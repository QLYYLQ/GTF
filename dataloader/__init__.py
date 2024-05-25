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
    path = os.path.join(Path(__file__).reslove().parent.parent, "dataset", "test", name)
    return path


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
