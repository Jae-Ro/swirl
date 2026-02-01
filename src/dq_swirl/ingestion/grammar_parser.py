import json
import re
from typing import Any, Dict, Optional, Tuple

from lark import Lark, Transformer, v_args

DEFAULT_GRAMMAR = r"""
    pair: KEY DELIMITER [VALUE]
    KEY: /[a-zA-Z_]\w*/
    DELIMITER: /\s*[:=]\s*/ | /\s+/
    VALUE: /.+?(?=(?:,\s*|\s+)[a-zA-Z_]\w*\s*[:=]|$)/
    %import common.WS
    %ignore WS
"""


class KeyValueTransformer(Transformer):
    @v_args(inline=True)
    def pair(self, key: str, delimiter: str, value: str = None) -> Tuple[str, Any]:
        """Method to transform a Lark 'pair' match into a clean key-value tuple

        :param key: identified key from the string segment
        :param delimiter: separator found (e.g., ':', '=', or whitespace)
        :param value: content following the delimiter, defaults to None
        :return: tuple containing the string key and its cleaned value or "None"
        """
        k = str(key).lower()
        v = "None"
        if value:
            v = str(value).rstrip(",").strip()
        return k, v


class GrammarParser:
    def __init__(self, grammar_override: Optional[str] = None) -> None:
        """Init method for GrammarParser

        :param grammar_override: pass in your own grammar rules string, defaults to None
        """
        self.pair_grammar = grammar_override or DEFAULT_GRAMMAR
        self.pair_parser = Lark(self.pair_grammar, start="pair", parser="earley")
        self.transformer = KeyValueTransformer()

        # pre-compile regex
        self.header_fix = re.compile(r"([a-zA-Z]+)\s+(\d+):\s*")
        self.splitter = re.compile(r"(?:,\s*|\s+)(?=[a-zA-Z_]\w*\s*[:=])")

    def _try_parse_json(self, trimmed: str) -> Optional[Dict[str, Any]]:
        """Helper function to parse stringified JSON objects/arrays

        :param trimmed: stringified JSON
        :return: python dictionary representation from `json.loads()`
        """
        if not (
            (trimmed.startswith("{") and trimmed.endswith("}"))
            or (trimmed.startswith("[") and trimmed.endswith("]"))
        ):
            return None

        try:
            data = json.loads(trimmed)
            if isinstance(data, dict):
                # normalize keys to lowercase and values to strings to match Lark output
                return {str(k).lower(): str(v) for k, v in data.items()}
            elif isinstance(data, list):
                # wrap arrays to maintain a KV structure
                return {"json_data": trimmed}
        except (json.JSONDecodeError, TypeError):
            return None
        return None

    def smart_parse(self, raw_str: str) -> Dict[str, Any]:
        """Entrypoint method for string parsing.

        Assumptions:
        * there exists some key value pair structure in the string.

        Primary Functions:
        * normalizes headers
        * splits line into logical segments
        * attempts to extract key-value pairs

        :param raw_str: The raw string to be processed
        :return: dictionary of extracted key-value pairs, plus an '_unparsed'
            field for any data that didn't match the key, value pair schema
        """
        trimmed = raw_str.strip()

        # json
        json_result = self._try_parse_json(trimmed)
        if json_result is not None:
            return json_result

        # key value pair string
        extracted = {}
        unparsed_segments = []

        # normalize headers
        content = self.header_fix.sub(r"\1=\2, ", raw_str)
        # split segments
        segments = self.splitter.split(content)

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            try:
                # k,v pairs
                tree = self.pair_parser.parse(seg)
                k, v = self.transformer.transform(tree)
                extracted[k] = v
            except Exception:
                # unparsed segments
                unparsed_segments.append(seg)

        if unparsed_segments:
            extracted["_unparsed"] = " ".join(unparsed_segments)

        return extracted
