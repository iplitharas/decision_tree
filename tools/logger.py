import logging
import sys
import time
import inspect
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_dir)


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper()


@singleton
class Logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(filename=os.path.join(parent_path, "logs.log"), mode="w")
        formatter = logging.Formatter(
            '[%(asctime)-10s:%(filename)s:%(lineno)d][%(levelname)s] >> %(message)s')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)


def logged(function):
    def wrapper(*args, **kwargs):
        Logger.logger.info(f"Calling {function.__name__}")
        start = time.time()
        result = function(*args, **kwargs)
        duration = time.time() - start
        if not inspect.isgeneratorfunction(function):
            Logger.logger.info(f"Called {function.__name__} for {duration:0.2f}s")
        else:
            Logger.logger.info(f"Called {function.__name__} (is a generator)")
        return result

    return wrapper
