import pytest
import logging

@pytest.fixture(autouse=True)
def configure_loggers(tmp_path):
    log_file = tmp_path / "tests.log"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    formatter = logging.Formatter("%(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    loggers = ["netexplainer", "scraper", "dataset", "llm"]
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    yield
