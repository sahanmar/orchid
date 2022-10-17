from typing import Dict, Any


def overwrite_config(dataclass_2_create):
    def load_overwritten_config(
        config: Dict[str, Any], overwrite: Dict[str, Any]
    ):
        unknown_keys = set(overwrite.keys()) - set(config.keys())
        if unknown_keys:
            raise ValueError(f"Unexpected config keys: {unknown_keys}")
        return dataclass_2_create(
            **{key: overwrite.get(key, val) for key, val in config.items()}
        )

    return load_overwritten_config
