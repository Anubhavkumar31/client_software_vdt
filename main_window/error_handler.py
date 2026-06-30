import logging
import traceback
import os
from PyQt6.QtWidgets import QMessageBox
from main_window.debug_config import DEBUG

import sys
import os

def get_app_dir():
    if getattr(sys, 'frozen', False):
        # running as exe
        return os.path.dirname(sys.executable)
    else:
        # running as normal python
        return os.path.dirname(os.path.abspath(__file__))

LOG_PATH = os.path.join(get_app_dir(), "app.log")

# create a dedicated logger just for your app, not root logger
logger = logging.getLogger("vdt_app")
logger.setLevel(logging.ERROR)  # only ERROR and above, no debug noise

handler = logging.FileHandler(LOG_PATH)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s\n%(message)s\n---"))
logger.addHandler(handler)

def handle_error(parent, e, user_msg="An error occurred"):
    logger.error(traceback.format_exc())  # only your errors go here
    if DEBUG:
        raise e
    else:
        QMessageBox.critical(parent, "Error", user_msg)