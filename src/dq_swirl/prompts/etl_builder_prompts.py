ARCHITECT_PROMPT = """You are a Lead Data Architect.
Define a simple Pydantic v2 `BaseModel` that represents the "Gold Standard" foundation for the data pattern found in the input samples.

INPUT SAMPLES (Multiple variations):
{samples}

EXECUTION FEEDBACK:
{feedback}

REQUIREMENTS:
1. Normalization: Suggest clean, snake_case keys for the identified fields.
2. Determine what fields should be required vs optional based on overall semantic meaning of the entity you are creating a BaseModel class for.
3. Include a detailed description for each field using the `Field` class to explain what the field is and if there are any expected structural patterns (e.g., `state` should be two letters).
4. Do NOT include any regex.
5. Attempt to capture all of the nuanced features from the sample data into your model as fields (e.g., special feature tags that exist within strings can be extracted as a separate field for the model).
6. You MUST wrap your code in a python block with the following start marking "```python" and end marking "```".
7. Create supplemental BaseModel classes where necessary to preserve semantic clarity.
8. You are ONLY allowed to use the following imports: "from typing import List, Dict, Optional; from pydantic import BaseModel, Field".
9. Keep primary keys as type string.
10. Infer best data type from string value (e.g., money should be a float, "true/false" or "yes/no" fields should be a boolean, and fields that represent multiple entities should use a representative aggregate data structure type)
11. NEVER set potentially boolean fields as optional. Instead, when not explicitly declared, infer as to what the default value ought based on the semantic meaning of the field and how it appears in the samples that do provide it.
12. Perform semantic merging: Identify fields across structural variants that share the same intent and conjoin them under a single, definitive schema key to avoid redundancy (e.g., "location" vs "city", "state", "zip code")
13. Avoid information loss when it comes to key:value pairs in the sample data.
14. If [EXECUTION FEEDBACK] is not "N/A," treat it as a diagnostic: analyze the underlying logic failure and re-architect the design to solve the root cause rather than just patching the symptom.

Return ONLY the Python code for the class. Include necessary imports (from pydantic import BaseModel, Field, etc.).
"""

CODER_PROMPT = """You are a Senior Data Engineer.
Your task is to write a concise but effective transformation function `transform_to_models(parsed_dict: list[dict]) -> list[dict]` that maps roughly parsed dictionaries into the provided pydantic v2 target schema base model definition.

TARGET SCHEMA (Python Pydantic v2 BaseModel):
{schema}

SOURCE SAMPLES:
{samples}

EXECUTION FEEDBACK:
{feedback}

Logic Requirements:
1. Use a 'coalesce' approach: for each target field, check all possible source keys from the input dictionary samples.
2. The Target Schema is the gold standard so ensure that the transformation function maps and casts the data types of the input appropriately.
3. Use parsed_dict.get() for optional fields.
4. Infer best data type from string (e.g., "$120.00" should be a float, and "true" should be a boolean). 
5. ALL python code must be encapsulated by the `transform_to_models()` function -- if it's not in that function it will not be run.
6. If [EXECUTION FEEDBACK] section is populated and is not "N/A", make sure to apply that feedback into your model generation process.

Return ONLY the Python code for the function `transform_to_models`. Do not include the Pydantic class in your response.
"""

CODE_EXECUTION_PROMPT = """
from pydantic import BaseModel, Field, ValidationError
from typing import *
import json, re

{schema}

{parser_code}
"""
