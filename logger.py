import pandas as pd
import torch
import os
import numpy as np
from collections import defaultdict
import time
import shutil

logger__ = None

def init_logger(logdir, logname, use_wandb=False):
    global logger__
    logger__ = Logger(logdir, logname, use_wandb)

def log():
    global logger__
    return logger__

class Logger():
    def __init__(self, logdir, logname, use_wandb=False):
        self.logdir = logdir
        self.use_wandb = use_wandb

        if logname.startswith("tmp") and os.path.exists(os.path.join(logdir, logname)):
            shutil.rmtree(os.path.join(logdir, logname))
        
        assert(os.path.isdir(logdir))
        self.dir = os.path.join(logdir, logname)
        os.mkdir(self.dir)

        self.params = dict()
        self.plots = dict()
        self.plots_columns = dict()

        self.time_metrics = defaultdict(float)
        self.prev_time = None

        if self.use_wandb:
            global wandb
            import wandb
            wandb.init(name=logname)

    def get_log_path(self):
        return self.dir



    def update_params(self, params):
        self.params.update(params.to_dict())
        params_path = os.path.join(self.dir, "params.csv")
        pd.DataFrame(self.params.items(), columns=("name", "value")).to_csv(params_path, index=False)
        if self.use_wandb:
            wandb.config.update(self.params)

    def add_plot(self, name, columns):
        assert name not in self.plots
        self.plots[name] = list()
        self.plots_columns[name] = columns

    def add_plot_point(self, name, point):
        self.plots[name].append(point)
        if self.use_wandb:
            wandb.log(dict(zip(self.plots_columns[name], point)))

    def add_plot_points(self, name, points):
        self.plots[name].extend(points)
        if self.use_wandb:
            wandb.log(dict(zip(self.plots_columns[name], zip(*points))))

    def get_plot(self, name):
        return self.plots[name]


    def check_time(self, add_to_value=None):
        now = time.time()
        if add_to_value:
            self.time_metrics[add_to_value] += now - self.prev_time
        self.prev_time = now
        return now

    def zero_time_metrics(self):
        self.time_metrics.clear()

    def print_time_metrics(self):
        print("=" * 50)
        print("="*18 + " Time metrics " + "=" * 18)
        for k, v in self.time_metrics.items():
            print("{}: {} seconds".format(k, v))

        print("=" * 50)

    # clears saved logs
    def save_logs(self):
        self.save_csv()
        self.clear_logs()

    def save_model(self, model, name):
        models_path = os.path.join(self.dir, "models")
        os.makedirs(models_path, exist_ok=True)
        torch.save(model, os.path.join(models_path, name))

    def save_csv(self):
        plot_path = os.path.join(self.dir, "plots")
        os.makedirs(plot_path, exist_ok=True)
        for plot_name, plot_data in self.plots.items():
            filename = os.path.join(plot_path, plot_name + ".csv")
            pd.DataFrame(plot_data, columns=self.plots_columns[plot_name]).to_csv(filename, index=False, mode='a', \
                    header=not os.path.exists(filename)) # append

    def clear_logs(self):
        for key, value in self.plots.items():
            value.clear()
        self.time_metrics.clear()
