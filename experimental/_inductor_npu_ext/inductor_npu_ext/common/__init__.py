def _setup_logger(name):
    import logging
    import sys
    import os

    rank = os.getenv("RANK", "0")
    level = logging.DEBUG if os.getenv('TORCH_COMPILE_DEBUG', '0') == '1' else logging.ERROR

    default_logger = logging.getLogger(name)
    default_logger.setLevel(level)

    if not default_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        format_str = f'[%(asctime)s] [RANK{rank}] [%(levelname)s] %(message)s'
        formatter = logging.Formatter(
            format_str,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        default_logger.addHandler(handler)
        default_logger.propagate = False

    return default_logger


logger = _setup_logger("inductor_asc")
