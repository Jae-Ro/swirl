import logging


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors to the level while graying out metadata"""

    # color codes
    grey = "\x1b[38;5;245m"
    cyan = "\x1b[36;20m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def get_format(self, level_color):
        return (
            f"{self.grey}%(asctime)s{self.reset} "
            f"{self.cyan}%(name)s{self.reset} "
            f"[{level_color}%(levelname)s{self.reset}] "
            f"{self.grey}%(filename)s:%(lineno)s{self.reset} "
            f"%(message)s"
        )

    def format(self, record):
        formats = {
            logging.DEBUG: self.get_format(self.blue),
            logging.INFO: self.get_format(self.green),
            logging.WARNING: self.get_format(self.yellow),
            logging.ERROR: self.get_format(self.red),
            logging.CRITICAL: self.get_format(self.bold_red),
        }

        log_fmt = formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_custom_logger(
    name: str = "swirl",
    log_level: int = logging.DEBUG,
) -> logging.Logger:
    """Function to return a custom formatter logger object

    :param name: name of logger, defaults to "swirl"
    :param log_level: desired logging level, defaults to logging.DEBUG
    :return: custom formatted Logger instance with StreamHandler
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    if len(logger.handlers) > 0:
        return logger

    # handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # formatters
    formatter = ColorFormatter()
    stream_handler.setFormatter(formatter)

    stream_handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.handlers = [stream_handler]

    return logger
