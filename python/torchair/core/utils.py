import logging
import sys


def _get_logger(*, level=logging.ERROR, output=sys.stdout, file=None, name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    try:
        import colorlog
    except ImportError:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            })

    if output:
        console_handler = logging.StreamHandler(output)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = _get_logger(name="torchair")
