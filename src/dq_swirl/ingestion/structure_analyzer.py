import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict


class Fingerprint(TypedDict):
    hash: str
    signature: Dict[str, Any]
    score: float
    full_str: str


class RawParsedPair(TypedDict):
    raw: str
    parsed: Dict[str, Any]


@dataclass(slots=True)
class SignatureEntry:
    signature: Dict[str, Any]
    records: List[RawParsedPair]


class StructuralAnalyzer:
    """Class to analyze dictionary structural signatures and hash them accordingly

    Signature Map Structure:
    ```python
    {
        <hash>: <SignatureEntry>
    }
    ```
    """

    def __init__(self, ignore_unparsed: bool = False):
        """Init Method

        :param ignore_unparsed: boolean flag to ignore the "_unparsed" field or not, defaults to False
        """
        self.signature_map: Dict[str, SignatureEntry] = {}
        self.ignore_unparsed = ignore_unparsed

    def _get_type(self, value: Any) -> str:
        """Method to get value type from key:value in dictionary

        :param value: value object
        :return: string representation/name of the value data type
        """
        if isinstance(value, dict):
            return "map"
        if isinstance(value, list):
            if len(value) > 0:
                # first element is a dict, drill down
                if isinstance(value[0], dict):
                    return "list[map]"
                return f"list[{type(value[0]).__name__}]"
            # list is empty
            return "list[empty_null]"
        return type(value).__name__

    def get_signature_map(self) -> Dict[str, SignatureEntry]:
        """Getter method for signature map

        :return: self.signature_map
        """
        return self.signature_map

    def flatten_and_type(
        self,
        data: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Recursive method flatten a nested dictionary into a dot-notated map of paths to data types

        :param data: input dictionary to analyze and flatten
        :param prefix: current dot-notated key path used during recursion, defaults to ""
        :return: dictionary where keys are dot-notated paths and values are string represented data types
        """
        items = {}
        for k, v in data.items():
            k_lower = k.lower()

            # handle ignore_unparsed
            if self.ignore_unparsed and k_lower == "_unparsed":
                continue

            key_path = f"{prefix}.{k_lower}" if prefix else k_lower

            # nested dict
            if isinstance(v, dict):
                # recurse
                items.update(self.flatten_and_type(v, key_path))
            # list
            elif isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], dict):
                    # recurse
                    items.update(self.flatten_and_type(v[0], f"{key_path}[]"))
                else:
                    items[key_path] = self._get_type(v)
            # empty list
            elif isinstance(v, list) and len(v) == 0:
                items[key_path] = "list[empty_null]"
            # primtive dtypes
            else:
                items[key_path] = self._get_type(v)

        return items

    def get_parseability(self, raw_str: str, flattened_data: Dict[str, Any]) -> float:
        """Method to calculate ratio of structured data vs. total input.

        :param raw_str: original raw string
        :param flattened_data: output of `flatten_and_type()`
        :return: score between 0.0 and 1.0 representing the data coverage score.
        """
        # use values from the flattened data to measure coverage
        structured_len = sum(len(str(v)) for v in flattened_data.values())
        total_len = len(raw_str)

        if total_len == 0:
            return 0.0

        # 2 decimal palces
        return round(min(structured_len / total_len, 1.0), 2)

    def generate_fingerprint(
        self,
        raw_input: str,
        parsed_dict: Dict[str, Any],
        store_in_map: bool = True,
    ) -> Fingerprint:
        """Method to generate a deterministic hash based on structure, while tracking parseability.

        :param raw_input: original raw string input
        :param parsed_dict: dictionary object to fingerprint
        :param store_in_map: boolean flat to store in `StructureAnalyzer.signature_map`, defaults to True
        :return: dictionary containing the hash, typed signature, parseability score,
            and summary string of the input structure
        """
        # map paths to types
        typed_map = self.flatten_and_type(parsed_dict)

        # blueprint structure string
        sorted_keys = sorted(typed_map.keys())
        blueprint_str = "|".join([f"{k}:{typed_map[k]}" for k in sorted_keys])

        # structure hash
        struct_hash = hashlib.md5(blueprint_str.encode()).hexdigest()

        # parseability
        score = self.get_parseability(raw_input, typed_map)

        # full string = blueprint + parseability
        # can use for clustering
        full_str = f"{blueprint_str}|parse_score:{score}"

        if store_in_map:
            new_record: RawParsedPair = {
                "raw": raw_input,
                "parsed": parsed_dict,
            }
            if struct_hash not in self.signature_map:
                self.signature_map[struct_hash] = SignatureEntry(
                    signature=typed_map,
                    records=[new_record],
                )
            else:
                self.signature_map[struct_hash].records.append(new_record)

        fingerprint: Fingerprint = {
            "hash": struct_hash,
            "signature": typed_map,
            "score": score,
            "full_str": full_str,
        }

        return fingerprint
