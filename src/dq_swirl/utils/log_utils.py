
import logging


def get_custom_logger(
    name: str = "dq-agent",
    log_level: int = logging.DEBUG
) -> logging.Logger:
    """Function to return a custom formatter logger object

    :param name: name of logger, defaults to "dq-agent"
    :param log_level: desired logging level, defaults to logging.DEBUG
    :return: custom formatted Logger instance with StreamHandler
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    if len(logger.handlers > 0):
        return logger
    
    # handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # formatters
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(filename)s:%(lineno)s [%(levelname)s]: %(message)s",
        "%Y-%m-%d %H:%M:%S" 
    )

    stream_handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.handlers = [stream_handler]

    return logger

