import logging
import sys
from pathlib import Path
from netexplainer.logger import configure_logger
from unittest.mock import patch

def test_logger_has_correct_name(tmp_path):
    logger_name = "test_logger_has_correct_name"
    logger = configure_logger(logger_name, tmp_path / "test.log")
    assert logger.name == logger_name

def test_logger_level_default(tmp_path):
    logger_name = "test_logger_level_default"
    logger = configure_logger(logger_name, tmp_path / "test.log")
    assert logger.level == logging.DEBUG

def test_logger_level_custom(tmp_path):
    logger_name = "test_logger_level_custom"
    logger = configure_logger(logger_name, tmp_path / "test.log", level=logging.WARNING)
    assert logger.level == logging.WARNING

def test_logger_has_file_and_console_handlers(tmp_path):
    logger_name = "test_logger_has_file_and_console_handlers"
    logger = configure_logger(logger_name, tmp_path / "test.log")
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    console_handlers = [
        h for h in logger.handlers 
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
    assert len(console_handlers) == 1

def test_handlers_have_correct_levels_and_formatters(tmp_path):
    logger_name = "test_handlers_have_correct_levels_and_formatters"
    level = logging.DEBUG
    logger = configure_logger(logger_name, tmp_path / "test.log", level=level)
    for handler in logger.handlers:
        assert handler.level == level
        assert isinstance(handler.formatter, logging.Formatter)
        assert handler.formatter._fmt == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert handler.formatter.datefmt == "%Y-%m-%d %H:%M:%S"

def test_log_directory_created(tmp_path):
    logger_name = "test_log_directory_created"
    log_dir = tmp_path / "subdir"
    log_file = log_dir / "test.log"
    configure_logger(logger_name, log_file)
    assert log_dir.exists()

def test_error_creating_directory_logs_error(tmp_path, caplog):
    logger_name = "test_error_creating_directory_logs_error"
    caplog.set_level(logging.ERROR)
    with patch.object(Path, "mkdir", side_effect=Exception("Mocked error")):
        log_file = tmp_path / "test.log"
        configure_logger(logger_name, log_file)
    assert "Error creating log directory: Mocked error" in caplog.text

def test_handlers_not_added_twice(tmp_path):
    logger_name = "test_handlers_not_added_twice"
    logger = configure_logger(logger_name, tmp_path / "test.log")
    initial_handlers = len(logger.handlers)
    logger = configure_logger(logger_name, tmp_path / "test.log")
    assert len(logger.handlers) == initial_handlers

def test_logger_propagate_is_false(tmp_path):
    logger_name = "test_logger_propagate_is_false"
    logger = configure_logger(logger_name, tmp_path / "test.log")
    assert logger.propagate is False

def test_file_handler_has_utf8_encoding(tmp_path):
    logger_name = "test_file_handler_has_utf8_encoding"
    logger = configure_logger(logger_name, tmp_path / "test.log")
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert file_handlers[0].encoding == "utf-8"

def test_console_handler_uses_stdout(tmp_path):
    logger_name = "test_console_handler_uses_stdout"
    logger = configure_logger(logger_name, tmp_path / "test.log")
    console_handlers = [
        h for h in logger.handlers 
        if isinstance(h, logging.StreamHandler) 
        and h.stream == sys.stdout
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].stream == sys.stdout
