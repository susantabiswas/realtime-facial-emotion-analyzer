# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Custom logger factory with a deafult logger
mathod to create a logger with file and console stream."""
# ===================================================
import logging
import sys
import traceback
from typing import List


class LoggerFactory:
    """Custom logger factory with a deafult logger
    method to create a logger with file and console stream.
    """

    def __init__(self, logger_name: str):
        """Creates a logger with default settings.

        Args:
            logger_name (str): logger name
        """
        self.logger = self.create_logger(logger_name=logger_name)

    def create_formatter(self, format_pattern: str):
        """Creates a logger formatter with user defined/deafult format.

        Args:
            format_pattern (str, optional): Logger message format. Defaults to None.
        Returns:
            logger formatter
        """
        format_pattern = (
            format_pattern
            or "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        return logging.Formatter(format_pattern)

    def get_console_handler(self, formatter, level=logging.INFO, stream=sys.stdout):
        """Returns a stream handler for logger

        Args:
            formatter : logger formatter object
            level (optional): Logger level. Defaults to logging.INFO.
            stream (stream, optional): Stream type. E.g: STDERR, STDOUT
                Defaults to sys.stdout.

        Returns:
            logger stream handler
        """
        # create a stream handler, it can be for stdout or stderr
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        return console_handler

    def get_file_handler(
        self, formatter, level=logging.INFO, file_path: str = "data/app.log"
    ):
        """Returns a file handler for logger

        Args:
            formatter : logger formatter object
            level (optional): Logger level. Defaults to logging.INFO.
            file_path (str, optional): Path where the log file should be saved.
                Defaults to 'data/app.log'.

        Returns:
            logger file handler
        """
        file_handler = logging.FileHandler(filename=file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        return file_handler

    def create_logger(
        self,
        logger_name,
        level=logging.DEBUG,
        format_pattern: str = None,
        file_path: str = "data/app.log",
    ):
        """Creates a logger with pre-defined settings

        Args:
            logger_name (str): Name of logger
            level (optional): Logger level. Defaults to logging.DEBUG.
            format_pattern (str, optional): Logger message format. Defaults to None.
            file_path (str, optional): Path where the log file should be saved.
                Defaults to 'data/app.log'.

        Returns:
            logger
        """
        # Creates the default logger
        logger = logging.getLogger(logger_name)
        formatter = self.create_formatter(format_pattern=format_pattern)
        # Get the stream handlers, by default they are set at INFO level
        console_handler = self.get_console_handler(
            formatter=formatter, level=logging.INFO
        )
        file_handler = self.get_file_handler(
            formatter=formatter, level=logging.INFO, file_path=file_path
        )
        # Add all the stream handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        # Set the logger level, this is different from stream handler level
        # Stream handler levels are further filters when data reaches them
        logger.setLevel(level)
        logger.propagate = False
        return logger

    def get_logger(self):
        """Returns the created logger
        Returns:
            logger
        """
        return self.logger

    def create_custom_logger(
        self, logger_name: str, handlers: List, propagate_error: bool = False
    ):
        """Creates a custom logger.

        Args:
            logger_name (str): Name of logger
            handlers (List) : Logger handlers. E.g: Stream handlers like file handler, console handler etc.
            propagate_error (bool, optional): Whether the errors should be propagated to the parent.
                Defaults to False.

        Returns:
            logger
        """
        logger = logging.getLogger(logger_name)
        # Add all the stream handlers
        for handler in handlers:
            logger.addHandler(handler)
        logger.propagate = propagate_error
        return logger

    def uncaught_exception_hook(self, type, value, tb):
        """Handles uncaught exceptions and saves the details
        using the logger. So if the logger has console and file
        stream handlers, then the uncaught exception will be sent there.

        Args:
            logger (logger): Python native logger
            type (Exception type):
            value (Exception value): Exception message
            tb (traceback):

        """
        # Returns a list of string sentences
        tb_message = traceback.extract_tb(tb).format()
        tb_message = "\n".join(tb_message)
        err_message = "Uncaught Exception raised! \n{}: {}\nMessage: {}".format(
            type, value, tb_message
        )
        self.logger.critical(err_message)

