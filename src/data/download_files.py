"""
Module responsible for downloading data files.
"""
import yaml
import gdown


def download_files():
    """
    Downloads three data files (train, test, validation) based on the specified
    links given in `config.yaml`.
    """

    with open('./config.yaml', 'r', encoding="UTF-8") as file:
        try:
            config = yaml.safe_load(file)
            for data in ["train", "test", "validation"]:
                gdown.download(config[f"{data}_data"], f"./data/{data}.tsv",
                               quiet=True, fuzzy=True)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == '__main__':
    download_files()
