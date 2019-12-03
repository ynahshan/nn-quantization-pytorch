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
home = str(Path.home())


def exit(trucker):
    trucker.__exit__()


class StatsTrucker(metaclass=Singleton):
    def __init__(self):
        self.folder = 'mxt-sim/stats'
        self.fname = 'stats.csv'
        self.mode = 'mean'
        self.stats = {}

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
        print('ho')
        # Save measures
        location = os.path.join(home, self.folder)
        if os.path.exists(location):
            shutil.rmtree(location)
        if not os.path.exists(location):
            os.makedirs(location)
        path = os.path.join(location, 'stats.csv')
        # for s in self.stats:
        #     df = pd.DataFrame(columns=self.stats_names, data=self.stats[s_id])
        #
        # df.to_csv(path, index=False)
