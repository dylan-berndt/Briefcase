import json


class Config:
    def __init__(self, **kwargs):
        object.__setattr__(self, "_values", kwargs)
        self._location = None

    def keys(self):
        return self._values.keys()

    def __getitem__(self, key):
        if "." in key:
            left, right = key.split(".")[0], ".".join(key.split(".")[1:])
            return self[left][right]

        return self._values[key]

    def __getattr__(self, key):
        if key.startswith("_"):
            return super().__getattribute__(key)
        if key in self._values:
            return self._values[key]

        raise AttributeError(f"{type(self).__name__} has no attribute {key}")

    def __setattr__(self, key, value):
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._values[key] = value

    def __setitem__(self, key, value):
        if key == "_values":
            object.__setattr__(self, key, value)
        else:
            self._values[key] = value

    def __iter__(self):
        return iter(self._values)

    def load(self, path):
        self._location = path
        with open(path, "r") as file:
            data = json.load(file)
            self._values = self._deserialize(data)._values

        return self
    
    def overwrite(self):
        self.save(self._location)

    def serialize(self):
        return self._serialize(self)

    def save(self, path):
        with open(path, "w+") as file:
            json.dump(self._serialize(self), file, indent=4)

    def items(self):
        return self._values.items()

    @staticmethod
    def _deserialize(data):
        if isinstance(data, dict):
            config = Config()
            for key, value in data.items():
                config._values[key] = Config._deserialize(value)
            return config
        return data

    @staticmethod
    def _serialize(data):
        if isinstance(data, Config) or isinstance(data, dict):
            return {k: Config._serialize(v) for k, v in data.items()}
        else:
            return data

def loadRunConfig(path, operational=None):
    """Run config for scripts that read their settings as flat attributes (finetuneTags.py, experiments/levjepa).
    The file is {"section": {key: value}, ...} plus optional top-level key: value pairs; every key is lifted to the
    top level (keys must be unique across sections). `operational` (an argparse.Namespace of flags that change how the
    process runs, not what it computes) is merged on top. The file as loaded is kept in `.sections` so a run can save
    exactly what defines it."""
    import argparse
    with open(path, encoding="utf8") as file:
        data = json.load(file)
    flat = {}
    for key, value in data.items():
        for innerKey, innerValue in (value.items() if isinstance(value, dict) else [(key, value)]):
            assert innerKey not in flat, f"{path}: config key {innerKey!r} appears twice"
            flat[innerKey] = innerValue
    for key, value in vars(operational or argparse.Namespace()).items():
        assert key not in flat, f"flag --{key} duplicates a key in {path}"
        flat[key] = value
    namespace = argparse.Namespace(**flat)
    namespace.sections = data
    namespace.configPath = path
    return namespace
