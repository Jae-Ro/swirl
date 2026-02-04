from __future__ import annotations

import json
import operator
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Dict, Optional, TypedDict

import virt_s3
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis

from dq_swirl.agents.etl_builder_agent import ETLBuilderAgent
from dq_swirl.agents.query_builder_agent import QueryBuilderAgent
from dq_swirl.clients.async_httpx_client import AsyncHttpxClient
from dq_swirl.clients.async_llm_client import AsyncLLMClient
from dq_swirl.ingestion.rust_ingestion import smart_parse_batch
from dq_swirl.ingestion.structure_analyzer import StructuralAnalyzer
from dq_swirl.ml_ai.clustering import ClusterOrchestrator
from dq_swirl.ml_ai.embedding_model import EmbeddingModel
from dq_swirl.persistence.signature_registry import SignatureRegistry
from dq_swirl.utils.agent_utils import (
    load_function,
    load_pydantic_base_models,
    prepause,
)
from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


class AgentOrchestratorState(TypedDict):
    run_id: str
    user_query: str
    request_config: Dict[str, str]
    data_key: str
    step_result: Annotated[Optional[str], lambda old, new: new]
    attempts: Annotated[int, operator.add]
    error: Optional[str]


class DQAgentOrchestrator:
    def __init__(
        self,
        client: AsyncLLMClient,
        redis: Optional[Redis] = None,
        s3_dirpath: str = "data/pipeline_runs",
        http_client: Optional[AsyncHttpxClient] = None,
        embedding_model: EmbeddingModel | str = "all-MiniLM-L6-v2",
        max_attempts: int = 3,
        sample_count: int = 100,
    ) -> None:
        self.client = client
        self.redis = redis
        self.s3_dirpath = s3_dirpath
        self.http_client = http_client
        self.max_attempts = max_attempts

        self.created_http_client = False
        if not self.http_client:
            self.http_client = AsyncHttpxClient()
            self.created_http_client = True

        self.etl_agent = ETLBuilderAgent(
            client=self.client,
            redis=self.redis,
            s3_dirpath=self.s3_dirpath,
            max_attempts=self.max_attempts,
        )

        self.clusterer = ClusterOrchestrator(embedding_model=embedding_model)
        self.analyzer = StructuralAnalyzer(ignore_unparsed=False)
        self.registry = SignatureRegistry(redis=redis)

        # build graph
        self.graph = self._build_graph()

        # number of times to sample
        self.sample_count = sample_count

    def create_run_id(self) -> str:
        # current datetime
        curr_dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"run_{curr_dt_str}"
        return run_id

    def _build_graph(self) -> StateGraph:
        """_summary_

        :return: _description_
        """
        ## Define Graph
        workflow = StateGraph(AgentOrchestratorState)

        # nodes
        workflow.add_node("data_sourcer", self.data_sourcer)
        workflow.add_node("etl_builder_agent", self.etl_builder_agent)
        workflow.add_node("etl_runner", self.etl_runner)
        workflow.add_node("query_builder_agent", self.query_builder_agent)

        # edges
        workflow.add_edge(START, "data_sourcer")

        workflow.add_conditional_edges(
            "data_sourcer",
            self.validate_sourcer,
            {
                "retry": "data_sourcer",
                "continue": "etl_builder_agent",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "etl_builder_agent",
            self.validate_etl_builder,
            {
                "retry": "etl_builder_agent",
                "continue": "etl_runner",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "etl_runner",
            self.validate_etl_runner,
            {
                "retry": "etl_builder_agent",
                "continue": "query_builder_agent",
                "end": END,
            },
        )

        workflow.add_conditional_edges(
            "query_builder_agent",
            self.validate_query,
            {
                "retry": "query_builder_agent",
                "end": END,
            },
        )

        return workflow.compile(checkpointer=MemorySaver())

    async def data_sourcer(self, state: AgentOrchestratorState) -> Dict[str, Any]:
        req_config = state["request_config"]
        data_key = state["data_key"]

        try:
            result = set()
            for _ in range(self.sample_count):
                try:
                    res = await self.http_client.request(
                        req_config["url"],
                        method=req_config["method"],
                        request_body=req_config["request_body"],
                    )
                    for rec_str in res[data_key]:
                        result.add(rec_str)

                except Exception:
                    pass

            result = {
                data_key: list(result),
            }

            logger.debug(json.dumps(result, indent=4))

            return {
                "step_result": result,
                "error": None,
            }
        except Exception as e:
            logger.exception(e)
            err_msg = traceback.format_exc()
            return {
                "error": err_msg,
                "attempts": 1,
                "step_result": None,
            }

    async def validate_sourcer(self, state: AgentOrchestratorState):
        res = state["step_result"]
        err_msg = state["error"]
        attempts = state["attempts"]
        data_key = state["data_key"]

        # data sourcer did not work
        if not res and err_msg:
            if attempts > self.max_attempts:
                return "end"
            return "retry"
        elif res and not err_msg:
            if len(res[data_key]) < 1:
                return "retry"

        # it worked
        return "continue"

    @prepause
    async def etl_builder_agent(self, state: AgentOrchestratorState):
        try:
            run_id = state["run_id"]
            data = state["step_result"]
            data_key = state["data_key"]
            attempts = state["attempts"]

            for i, payload in enumerate(data[data_key]):
                if not isinstance(payload, str):
                    payload = f"{payload}"
                    data[data_key][i] = payload

            # run smart parse batch
            data_samples = smart_parse_batch(data[data_key])

            # run analyzer
            result = []
            for raw, parsed in data_samples:
                logger.debug(f"RAW: {raw}, PARSED: {parsed}")
                fingerprint = self.analyzer.generate_fingerprint(
                    raw,
                    parsed,
                    store_in_map=True,
                )
                hash_sign = fingerprint["hash"]
                result.append((raw, parsed, hash_sign))

            # determine if hashes live in redis signature registry
            curr_map = self.analyzer.get_signature_map()
            logger.debug(json.dumps(curr_map, indent=4))
            new_signs_exist = False
            for sign, sign_entry in curr_map.items():
                res = await self.registry.lookup_hash_signature(sign)
                if not res:
                    new_signs_exist = True

            # continue early
            if not new_signs_exist:
                return {
                    "step_result": result,
                    "error": None,
                    "attempts": 1,
                }

            # if there are new signatures, run clustering
            cluster_map = self.clusterer.make_clusters(curr_map)
            export_map, cluster_sets = await self.etl_agent.run(
                cluster_map,
                run_id=run_id,
            )
            logger.debug(f"ETL Lookup Map:\n{json.dumps(export_map, indent=4)}")
            logger.debug(f"Cluster Sets:\n{json.dumps(cluster_sets, indent=4)}")

            return {
                "step_result": result,
                "error": None,
                "attempts": 1,
            }

        except Exception as e:
            err_msg = traceback.format_exc()
            logger.exception(e)
            return {
                "error": err_msg,
                "attempts": 1,
                "step_result": None,
            }

    async def validate_etl_builder(self, state: AgentOrchestratorState):
        res = state["step_result"]
        err_msg = state["error"]
        attempts = state["attempts"]

        # etl builder did not work
        if not res and err_msg:
            if attempts > self.max_attempts:
                return "end"
            return "retry"

        # it worked
        return "continue"

    async def etl_runner(self, state: AgentOrchestratorState):
        try:
            result_samples = state["step_result"]
            s3_params = virt_s3.get_default_params()

            curr_dir = os.getcwd()

            # group by base_model, parser
            etl_dict = {}

            with virt_s3.SessionManager(params=s3_params) as session:
                for raw, parsed, hash_sign in result_samples:
                    sign_metadata = await self.registry.lookup_hash_signature(hash_sign)
                    base_model_fpath = sign_metadata.base_model_fpath.lstrip("/")
                    parser_fpath = sign_metadata.parser_fpath.lstrip("/")

                    # local file paths
                    base_model_local_fpath = os.path.join(curr_dir, base_model_fpath)
                    parser_local_fpath = os.path.join(curr_dir, parser_fpath)

                    # create dirs locally
                    Path(base_model_local_fpath).parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    Path(parser_local_fpath).parent.mkdir(parents=True, exist_ok=True)

                    # get data from s3 to local fpath
                    if not os.path.exists(base_model_fpath):
                        virt_s3.get_file(
                            base_model_fpath,
                            base_model_local_fpath,
                            params=s3_params,
                            client=session,
                        )
                    if not os.path.exists(parser_local_fpath):
                        virt_s3.get_file(
                            parser_fpath,
                            parser_local_fpath,
                            params=s3_params,
                            client=session,
                        )

                    key = f"{base_model_fpath}, {parser_fpath}"
                    etl_dict[key] = etl_dict.get(
                        key,
                        {
                            "base_model": base_model_local_fpath,
                            "parser": parser_local_fpath,
                            "data": [],
                        },
                    )
                    etl_dict[key]["data"].append(parsed)

            # run etl scripts
            result = []
            for etl_key, etl_meta in etl_dict.items():
                data = etl_meta["data"]
                base_model_fpath = etl_meta["base_model"]
                parser_fpath = etl_meta["parser"]
                parse_func = load_function(parser_fpath, "transform_to_models")
                res_li = parse_func(data)
                for res in res_li:
                    result.append(res_li)

            logger.debug(f"PARSED DATA RESULT:\n{json.dumps(result, indent=4)}")

            return {
                "step_result": result,
                "attempt": 1,
                "error": None,
            }

        except Exception as e:
            logger.exception(e)
            err_msg = traceback.format_exc()

            return {
                "error": err_msg,
                "attempts": 1,
            }

    async def validate_etl_runner(self, state: AgentOrchestratorState):
        return "end"

    @prepause
    async def query_builder_agent(self, state: AgentOrchestratorState):
        pass

    async def validate_query(self, state: AgentOrchestratorState):
        pass

    async def run(
        self,
        request_config: Dict[str, str],
        user_query: str,
        data_key: str = "raw_orders",
    ) -> AsyncGenerator[str, None]:
        """_summary_

        :param request_config: _description_
        :param user_query: _description_
        :param data_key: _description_, defaults to "raw_orders"
        :return: _description_
        """

        run_id = self.create_run_id()
        initial_state = {
            "run_id": run_id,
            "user_query": user_query,
            "request_config": request_config,
            "data_key": data_key,
        }
        config = {
            "configurable": {
                "thread_id": f"DQ-{run_id}",
            }
        }

        final_output = await self.graph.ainvoke(initial_state, config)

        # cleanup
        if self.created_http_client:
            await self.http_client.aclose()
