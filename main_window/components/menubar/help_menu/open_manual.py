import os
import sys


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def open_manual(self):
    p = resource_path(os.path.join("manual", "user_manual.pdf"))
    if os.path.exists(p):
        os.startfile(p)
    else:
        self.open_Error("User manual is not found.")