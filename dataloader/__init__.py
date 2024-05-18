import os
from pathlib import Path

def get_datapath(name):
    path = os.path.join(Path(__file__).resolve().parent.parent,"dataset",name)
    return path


if __name__ == "__main__":
    print(get_datapath("kaist"))

