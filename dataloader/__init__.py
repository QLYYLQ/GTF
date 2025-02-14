import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


import os
from pathlib import Path
from dataloader.kaist import kaistdataset
from dataloader.TNO import TNOdataset
from dataloader.RoadScene import RoadScenedataset
from dataloader.VIFB import VIFBdataset


def get_datapath(name):
    path = os.path.join(Path(__file__).resolve().parent.parent, "dataset", name)
    return path


def get_testdatapath(name):
    path = os.path.join(Path(__file__).resolve().parent.parent, "dataset", "test", name)
    return path

def get_imgsize(name):
    return {
        "VIFB":(460,630),
        "RoadScene":(1343,1004),
        "TNO":(620,450)
    }[name]


def get_dataset(name):
    return {
        "kaist": kaistdataset,
        "TNO": TNOdataset,
        "RoadScene": RoadScenedataset,
        "VIFB": VIFBdataset,
    }[name]


if __name__ == "__main__":
    print(get_datapath("kaist"))
    a = get_dataset("kaist")(get_datapath("kaist"))
    print(a)
