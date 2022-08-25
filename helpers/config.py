import yaml


def get_config():
    with open('./config.yml', 'r') as ymlfile:
        return yaml.safe_load(ymlfile)
