from typing import Dict, Any, List, Optional


def overwrite_config(dataclass_2_create):  # type: ignore
    def load_overwritten_config(  # type: ignore
        config: Dict[str, Any], overwrite: Dict[str, Any]
    ):
        unknown_keys = set(extract_dict_keys(overwrite)) - set(
            extract_dict_keys(config)
        )
        if unknown_keys:
            raise ValueError(f"Unexpected config keys: {unknown_keys}")
        return dataclass_2_create(**overwrite_dict_keys(config, overwrite))

    return load_overwritten_config


def extract_dict_keys(d: Dict[str, Any]) -> List[Any]:
    keys: List[Any] = []
    for k, v in d.items():
        if isinstance(v, dict):
            keys.extend(extract_dict_keys(v))
        keys.append(k)
    return keys


def overwrite_dict_keys(
    orig: Dict[str, Any], overwrite: Dict[str, Any]
) -> Dict[str, Any]:
    updated_dict: Dict[str, Any] = {}
    for orig_key, orig_val in orig.items():
        overwrite_val = overwrite.get(orig_key, None)
        if overwrite_val is None:
            updated_dict[orig_key] = orig_val
        elif isinstance(overwrite_val, dict):
            updated_dict[orig_key] = overwrite_dict_keys(
                orig_val, overwrite_val
            )
        else:
            updated_dict[orig_key] = overwrite_val
    return updated_dict


def get_overwrite_value(conf_dict: Dict[str, Any], key: str) -> Dict[str, Any]:
    value: Optional[Dict[str, Any]] = conf_dict.get(key, None)
    if value is None:
        print(
            f"Empty dict to overwrite {key}, so lay back and chill. Unless you expect smth to be there\n"
        )
        return {}
    return value
