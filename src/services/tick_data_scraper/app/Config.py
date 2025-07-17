import yaml
class Config:
    def __init__(self, path="config.yaml"):
        with open(path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]