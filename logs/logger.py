import logging

logging.basicConfig(
    filename="app.log",
    filemode="w",
    format="%(asctime)s- %(level)s- %(message)",
    datefmt= "%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)