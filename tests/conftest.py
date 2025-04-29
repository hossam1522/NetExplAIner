import pytest
from pathlib import Path
import tempfile
import logging

@pytest.fixture(autouse=True)
def configure_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "tests.log"
        logger = logging.getLogger("netexplainer")
        logger.handlers.clear()

        formatter = logging.Formatter("%(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        yield
