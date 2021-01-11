import plotly as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
import os
import re
import math

import torch
import numpy as np
import pandas as pd
import json


def add_trace(plot, x, y, name, color=None):
    plot.add_trace(go.Scatter(x=x, y=y, name=name, line_color=color))

def add_avg_trace(plot, x, y, name, avg_epochs, color=None):
    add_trace(plot, x, make_smooth(y, avg_epochs), name, color=color)

def make_smooth(y, avg_epochs=1):
    ny = list()
    
    cur_count = 0
    cur_val = 0
    for i in range(len(y)):
        cur_count = min(cur_count + 1, avg_epochs)
        if i >= avg_epochs:
            cur_val -= y[i - avg_epochs]
        cur_val += y[i]

        ny.append(cur_val / cur_count)

    return ny

def add_vertical_line(plot, x, y_st, y_en, name, color=None):
    add_trace(plot, x=[x, x], y=[y_st, y_en], name=name, color=color)


def add_reward_trace(plot, plot_data, use_steps=True, avg_epochs=1, name="reward"):
    if use_steps:
        plot_data = plot_data[np.argsort(plot_data[:, 1])]
    else:
        plot_data = plot_data[np.argsort(plot_data[:, 0])]
    if not len(plot_data):
        return
    train_episodes, steps, rewards = zip(*plot_data)

    y = np.array(rewards)
    if use_steps:
        x = np.array(steps)
    else:
        x = np.array(train_episodes)
    add_avg_trace(plot, x, y, name=name, avg_epochs=avg_epochs)


def create_reward_plot(plot_data, title="reward plot", use_steps=False, avg_epochs=1):
    plot = go.Figure()
    plot.update_layout(title=title)
    add_reward_trace(plot, plot_data, use_steps, avg_epochs)
    return plot


def load_csv(path):
    dataframe = pd.read_csv(path, index_col=False)
    return dataframe.to_numpy()

def get_last_log(logdir):
    return max([os.path.join(logdir, d) for d in os.listdir(logdir)], key=os.path.getmtime)


def get_paths(dir, prefix=""):
    return sorted(filter(lambda path: path.startswith(prefix), os.listdir(dir)),
            key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

def add_rewards(plot, logpath, use_steps=True, avg_constant=20, transform=lambda x: x, name="reward", env=-1):
    dirpath = logpath

    paths = get_paths(os.path.join(dirpath, "plots"), name)
    if not paths:
        paths = get_paths(os.path.join(dirpath, "plots"))

    for path in paths:
        data = load_csv(os.path.join(logpath, "plots", path))
        if env == -1:
            data = data[:, 0:3]
        else:
            mask = (data[:, 3] == env)
            data = data[:, 0:3][mask]
        if name == "time":
            data[:, 2] -= data[0][2]
        data[:, 2] = transform(data[:, 2])
        add_reward_trace(plot, data, use_steps=use_steps, avg_epochs=avg_constant,
                name=logpath[len("logdir"):])# + path[:-len(".csv")]) # ugly yeah

    return plot


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logpath', type=str, default=None, required=False)
    parser.add_argument('--name', type=str, default="reward", required=False)
    parser.add_argument('--avg', type=int, default=100, required=False)
    parser.add_argument('--env', type=int, default=-1, required=False)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args() 
    if args.logpath is None:
        logpaths = [get_last_log("logdir")]
    else:
        logpaths = list(filter(lambda x: re.fullmatch(args.logpath, x), os.listdir("logdir"))) 
        logpaths = [os.path.join("logdir", d) for d in logpaths]

    name = args.name

    
    rewards_plot = go.Figure()
    rewards_plot.update_layout(title=name, xaxis_title="Optimization step", yaxis_title=name)

    use_steps = (name == "time")
    transform = lambda data: data
    avg_constant = args.avg
    for logpath in logpaths:
        add_rewards(rewards_plot, logpath, use_steps=use_steps, avg_constant=avg_constant, transform=transform, name=name, env=args.env)

    plt.offline.plot(rewards_plot, filename="generated/rewards_plot.html")

