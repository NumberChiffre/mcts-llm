import logging

level = logging.INFO
logger = logging.getLogger("mcts-llm")
formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s|%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger.setLevel(level)

console_handler = logging.StreamHandler()
console_handler.setLevel(level)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
