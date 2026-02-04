from __future__ import annotations

import asyncio
import json
import operator
import os
import random
import traceback
from datetime import datetime
from functools import wraps
from io import BytesIO
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict

import virt_s3
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from litellm import ModelResponse
from pydantic import BaseModel, Field

from dq_swirl.clients.async_llm_client import AsyncLLMClient
from dq_swirl.ml_ai.clustering import ClusterRecord
from dq_swirl.persistence.signature_registry import ETLMap, SignatureRegistry
from dq_swirl.prompts.etl_builder_prompts import (
    ARCHITECT_PROMPT,
    CODE_EXECUTION_PROMPT,
    CODER_PROMPT,
)
from dq_swirl.utils.agent_utils import extract_python_code
from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


def prepause(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        await asyncio.sleep(random.uniform(0.2, 0.5))
        return await func(*args, **kwargs)

    return wrapper


class ModelResponseStructure(BaseModel):
    code_string: str = Field(
        ...,
        description="generated python code",
    )
    entrypoint_class_name: str = Field(
        ...,
        description="name of entrypoint base model class in the code generated",
    )


class MultiAgentState(TypedDict):
    semantic_cluster_id: str
    structure_cluster_id: str
    data_pairs_all: List[ClusterRecord]
    data_pairs_structure: List[ClusterRecord]
    gold_schema: Annotated[Optional[ModelResponseStructure], lambda old, new: new]
    parser_code: Annotated[Optional[str], lambda old, new: new]
    feedback: Annotated[Optional[str], lambda old, new: new]
    error_type: Annotated[
        Optional[Literal["SCHEMA_ISSUE", "CODE_ISSUE"]], lambda old, new: new
    ]
    attempts: Annotated[int, operator.add]  # increment
    export_map: Annotated[Optional[Dict[str, Any]], lambda old, new: new]


class ETLBuilderAgent:
    def __init__(
        self,
        client: AsyncLLMClient,
        redis_url: Optional[str] = None,
        s3_dirpath: str = "data/pipeline_runs",
        max_attempts: int = 6,
        max_sample_size: int = 100,
    ) -> None:
        """_summary_

        :param client: _description_
        :param redis_url: _description_, defaults to None
        :param s3_dirpath: _description_, defaults to "data/pipeline_runs"
        :param max_attempts: _description_, defaults to 6
        :param max_sample_size: _description_, defaults to 100
        """

        self.client = client
        self.max_attempts = max_attempts
        self.max_sample_size = max_sample_size
        self.redis_url = redis_url
        self.s3_dirpath = s3_dirpath

        # build graph at init
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """_summary_

        :return: _description_
        """
        ## Define Graph
        workflow = StateGraph(MultiAgentState)

        # nodes
        workflow.add_node("architect", self.architect_node)
        workflow.add_node("schema_tester", self.schema_tester_node)
        workflow.add_node("coder", self.coder_node)
        workflow.add_node("code_tester", self.code_tester_node)
        workflow.add_node("exporter", self.exporter_node)

        # edges
        workflow.add_edge(START, "architect")
        workflow.add_edge("architect", "schema_tester")
        workflow.add_conditional_edges(
            "schema_tester",
            self.schema_router,
            {
                "architect": "architect",
                "coder": "coder",
                "end": END,
            },
        )

        workflow.add_edge("coder", "code_tester")
        workflow.add_conditional_edges(
            "code_tester",
            self.code_router,
            {
                "coder": "coder",
                "exporter": "exporter",
                "end": END,
            },
        )

        workflow.add_edge("exporter", END)

        return workflow.compile(checkpointer=MemorySaver())

    @prepause
    async def architect_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """_summary_

        :param state: _description_
        :return: _description_
        """

        # base case
        if state.get("gold_schema") and state.get("error_type") != "SCHEMA_ISSUE":
            return {
                "attempts": 0,
            }

        logger.info(f"[Architect] Defining Semantic Goal. Attempt: {state['attempts']}")
        # grab state vars
        data_records = state["data_pairs_all"]
        feedback = state["feedback"]

        if feedback == "SUCCESS" or feedback is None:
            feedback = "N/A"

        # diversity is key to generalize
        sample_li = [p.parsed for p in data_records[: self.max_sample_size]]
        samples = json.dumps(sample_li, indent=2)
        logger.debug(f"Input SubSample: \n{json.dumps(sample_li[:3], indent=2)}")

        prompt = ARCHITECT_PROMPT.format(
            samples=samples,
            feedback=feedback,
        )
        buffer = []
        response = await self.client.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            temperature=0.0,
            response_format=ModelResponseStructure,
        )

        async for chunk in response:
            chunk: ModelResponse
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                buffer.append(content)

        resp = "".join(buffer)
        resp = ModelResponseStructure(**json.loads(resp))
        resp.code_string = extract_python_code(resp.code_string)
        logger.debug(f"Base Model Definition:\n{resp.code_string}")

        return {
            "gold_schema": resp,
            "attempts": 1,
            "feedback": None,
            "error_type": None,
        }

    async def schema_tester_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """_summary_

        :param state: _description_
        :return: _description_
        """
        logger.info(
            f"[Scehma Tester] Validating Functional BaseModel. Attempt: {state['attempts']}"
        )
        python_base_model_str = state["gold_schema"].code_string
        attempts = state["attempts"]

        env = {}
        try:
            exec(python_base_model_str, globals(), env)
            cls_name = state["gold_schema"].entrypoint_class_name
            model = env[cls_name]

            model.model_rebuild(_types_namespace=env)
            _ = model.model_json_schema()

            return {
                "feedback": "SUCCESS",
                "attempts": -1 * attempts,
            }

        except Exception as e:
            err_msg = traceback.format_exc()
            logger.exception(e)

            return {
                "feedback": err_msg,
                "error_type": "SCHEMA_ISSUE",
            }

    @prepause
    async def coder_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """_summary_

        :param state: _description_
        :return: _description_
        """
        logger.info(f"[Coder] Parser for Gold Schema: {state['attempts']}")

        # get state vars
        struct_records = state["data_pairs_structure"]
        feedback = state["feedback"]
        schema = state["gold_schema"].code_string

        if feedback == "SUCCESS" or feedback is None:
            feedback = "N/A"

        samples = json.dumps(
            [rec.parsed for rec in struct_records[: self.max_sample_size]],
            indent=2,
        )

        prompt = CODER_PROMPT.format(
            schema=schema,
            samples=samples,
            feedback=feedback,
        )
        buffer = []
        response = await self.client.chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            temperature=0.0,
        )
        async for chunk in response:
            chunk: ModelResponse
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                buffer.append(content)

        resp = "".join(buffer)
        code = extract_python_code(resp)

        return {
            "parser_code": code,
            "attempts": 1,
            "feedback": None,
            "error_type": None,
        }

    async def code_tester_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """_summary_

        :param state: _description_
        :return: _description_
        """
        logger.info(f"[Code Tester] Stress-testing parser: {state['attempts']}")

        schema = state["gold_schema"].code_string
        cls_name = state["gold_schema"].entrypoint_class_name
        parser_code = state["parser_code"]
        input_data_parsed = state["data_pairs_structure"]

        full_code = CODE_EXECUTION_PROMPT.format(
            schema=schema,
            parser_code=parser_code,
        )

        env = {}
        try:
            exec(full_code, globals(), env)
            func = env["transform_to_models"]
            model = env[cls_name]
            model.model_rebuild(_types_namespace=env)

            input_data = [pair.parsed for pair in input_data_parsed]
            mapped_batch = func(input_data)

            for mapped_dict in mapped_batch:
                model.model_validate(mapped_dict)
                logger.debug(f"PASSED: {mapped_dict}")

            logger.debug(f"Successful Parsing Code:\n{parser_code}")

            return {
                "feedback": "SUCCESS",
            }

        except Exception as e:
            try:
                logger.warning(f"FAILED: {mapped_dict}")
            except Exception:
                pass
            finally:
                err_msg = traceback.format_exc()
                logger.exception(e)

            return {
                "feedback": err_msg,
                "error_type": "CODE_ISSUE",
            }

    async def exporter_node(self, state: MultiAgentState) -> Dict[str, Any]:
        """_summary_

        :param state: _description_
        :return: _description_
        """

        # get necesary fields
        sem_id = state["semantic_cluster_id"]
        structure_cluster_id = state["structure_cluster_id"]
        data_pairs_structure = state["data_pairs_structure"]
        base_model_name = state["gold_schema"].entrypoint_class_name.lower()

        base_model_code_str = state["gold_schema"].code_string
        parser_code_str = state["parser_code"]
        export_map = state.get("export_map", {})

        # make the directory
        dir_name = self.s3_dirpath
        dir_name = os.path.join(
            dir_name,
            self.run_id,
            f"sem_{sem_id}-{base_model_name}",
        )

        # create bytesio objects to upload
        base_model_bytesio = BytesIO(base_model_code_str.encode("utf-8"))
        parser_bytesio = BytesIO(parser_code_str.encode("utf-8"))

        base_model_fpath = os.path.join(
            dir_name,
            f"{base_model_name}_base_model.py",
        )
        parser_fpath = os.path.join(
            dir_name,
            f"{base_model_name}_parser-struct_{structure_cluster_id}.py",
        )

        # upload to s3
        s3_params = virt_s3.get_default_params()
        with virt_s3.SessionManager(params=s3_params) as session:
            virt_s3.upload_data(
                base_model_bytesio,
                base_model_fpath,
                params=s3_params,
                client=session,
            )
            logger.info(f"Exported: {base_model_fpath}")
            virt_s3.upload_data(
                parser_bytesio,
                parser_fpath,
                params=s3_params,
                client=session,
            )
            logger.info(f"Exported: {parser_fpath}")

        # update export map
        seen = {}
        unique_struct_records = []
        for rec in data_pairs_structure:
            if rec.signature_hash not in seen:
                unique_struct_records.append(
                    {
                        "signature_hash": rec.signature_hash,
                        "fields": rec.fields,
                    }
                )
                seen[rec.signature_hash] = 1

        export_map[sem_id] = export_map.get(
            sem_id,
            {
                "base_model_fpath": base_model_fpath,
                "structure_clusters": [],
            },
        )
        export_map[sem_id]["structure_clusters"].append(
            {
                "id": structure_cluster_id,
                "parser_fpath": parser_fpath,
                "struct_records": unique_struct_records,
            }
        )

        return {
            "feedback": "DONE",
            "export_map": export_map,
        }

    def schema_router(self, state: MultiAgentState) -> str:
        """Router method for architect

        State Pathways:
        * schema_tester -> architect
        * schema_tester -> coder

        :param state: _description_
        :return: _description_
        """
        feedback = state.get("feedback")
        attempts = state.get("attempts", 0)

        if feedback == "SUCCESS":
            return "coder"

        # if failed too many times, just stop the process
        if attempts >= 3:
            logger.error(f"Schema failed after {attempts} attempts. Aborting.")
            return "end"

        return "architect"

    def code_router(self, state: MultiAgentState) -> str:
        """Router method for coder

        State Pathways:
        * code_tester -> coder
        * code_tester -> exporter

        :param state: _description_
        :return: _description_
        """
        feedback = state.get("feedback")
        attempts = state.get("attempts", 0)

        if feedback == "SUCCESS":
            return "exporter"

        if attempts >= 6:
            return "end"

        # default to retrying the coder for CODE_ISSUE or unknown errors
        return "coder"

    async def run(
        self,
        cluster_dict: Dict[str, List[ClusterRecord]],
        run_id: Optional[str] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, ETLMap]]:
        """_summary_

        :param cluster_dict: _description_
        :param run_id: _description_, defaults to None
        :return: _description_
        """
        if run_id is None:
            # current datetime
            curr_dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_id = f"run_{curr_dt_str}"

        shared_export_map = {}

        for sem_id, records in cluster_dict.items():
            struct_groups = {}
            for rec in records:
                cid = rec.structure_cluster_id
                struct_groups.setdefault(cid, []).append(rec)

            shared_gold_schema = None

            for struct_id, pairs in struct_groups.items():
                config = {
                    "configurable": {
                        "thread_id": f"sem_{sem_id}_str_{struct_id}",
                    }
                }

                initial_state = {
                    "semantic_cluster_id": str(sem_id),
                    "structure_cluster_id": str(struct_id),
                    "data_pairs_all": records,
                    "data_pairs_structure": pairs,
                    "gold_schema": shared_gold_schema,
                    "export_map": shared_export_map,
                    "feedback": None,
                    "attempts": 0,
                }

                final_output = await self.graph.ainvoke(initial_state, config)
                shared_gold_schema = final_output.get("gold_schema")
                shared_export_map = final_output.get("export_map")
                logger.info(
                    f"--- Finished Sem{sem_id}-Struct{struct_id} ---\n{json.dumps(shared_export_map, indent=4)}"
                )

        # format for lookup
        registry = SignatureRegistry(redis_url=self.redis_url)
        clusters, etl_map = registry.create_etl_lookup(shared_export_map)

        # store shared_export_map in redis for lookup if redis provided
        if self.redis_url:
            await registry.store_etl_lookup(
                clusters,
                etl_map,
            )
            await registry.close()

        return clusters, etl_map
