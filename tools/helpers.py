from tools.logger import Logger
import pickle


def write(file_path: str, data) -> None:
    Logger.logger.debug(f"Creating checkpoint at: {file_path}")
    with open(f'{file_path}.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def restore(file_path: str):
    Logger.logger.debug(f"Trying to restore checkpoint from:{file_path}")
    with open(f"{file_path}.pickle", 'rb') as f:
        data = pickle.load(f)
    return data
