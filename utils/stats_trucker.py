from utils.misc import Singleton
import numpy as np
import pandas as pd
import signal
import atexit
import os
import shutil
from utils.misc import sorted_nicely
import torch
from pathlib import Path
import pickle
home = str(Path.home())


def list_mean(l):
    if len(l) == 0:
        return None

    res = l[0].clone()
    for t in l[1:]:
        res += t
    return res / len(l)


def exit(trucker):
    trucker.__exit__()


class StatsTrucker(metaclass=Singleton):
    def __init__(self, sufix):
        self.folder = 'mxt-sim/stats'
        self.fname = 'stats_{}.pkl'.format(sufix)
        self.mode = 'mean'
        self.stats = {}
        self.exited = False

        signal.signal(signal.SIGINT, exit)
        signal.signal(signal.SIGTERM, exit)
        atexit.register(exit, self)

    def add(self, stat_name, id, value):
        if stat_name not in self.stats:
            self.stats[stat_name] = {}
        if id not in self.stats[stat_name]:
            self.stats[stat_name][id] = [value]
        else:
            self.stats[stat_name][id].append(value)

    def get_stats(self):
        stats = {}
        for s in self.stats:
            stats[s] = [list_mean(self.stats[s][k]).mean().item() for k in self.stats[s]]

        return stats

    def __exit__(self, *args):
        if self.exited:
            return
        print("Saving stats.")
        # Save measures
        location = os.path.join(home, self.folder)
        # if os.path.exists(location):
        #     shutil.rmtree(location)
        if not os.path.exists(location):
            os.makedirs(location)
        f = open(os.path.join(location, self.fname), 'wb')
        pickle.dump(self.stats, f)
        f.close()
        self.exited = True
