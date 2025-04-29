import pytest
import logging
from pathlib import Path
import tempfile

@pytest.fixture(autouse=True)
def configure_logging(tmp_path):
    log_file = tmp_path / "tests.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter("%(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    app_loggers = ["netexplainer", "scraper", "dataset", "llm"]
    for name in app_loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    yield

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()