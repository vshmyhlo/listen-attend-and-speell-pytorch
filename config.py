class Config(object):
    def __init__(self, config):
        if not isinstance(config, dict):
            raise ValueError('config must be dict')

        self.config = config

    def __getattr__(self, key):
        if key not in self.config:
            raise KeyError(key)

        value = self.config[key]

        if isinstance(value, dict):
            value = Config(value)

        return value

    @classmethod
    def from_yaml(cls, path):
        from ruamel.yaml import YAML

        with open(path) as f:
            return cls(YAML().load(f))

    @classmethod
    def from_json(cls, path):
        import json

        with open(path) as f:
            return cls(json.load(f))
