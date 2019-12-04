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


def exit(trucker):
    trucker.__exit__()


class StatsTrucker(metaclass=Singleton):
    def __init__(self):
        self.folder = 'mxt-sim/stats'
        self.fname = 'stats.csv'
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
        f = open(os.path.join(location, 'stats.pkl'), 'wb')
        pickle.dump(self.stats, f)
        f.close()
        self.exited = True
