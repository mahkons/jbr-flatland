import torch
from multiprocessing import Pool, set_start_method
from functools import partial
import argparse
from copy import deepcopy

from train import start_experiment

__device = None

def load_experiments(exp_path, sdevice):
    exp_list = torch.load(exp_path)
    
    # TODO a better way to choose cuda device for experiment
    # by the way only 4 experiments and 4 cuda is common case
    CUDA_COUNT = 4
    count = 0
    for exp in exp_list:
        if sdevice == "cpu":
            __device = torch.device("cpu")
        elif sdevice == "cuda":
            __device = torch.device("cuda:" + str(count % CUDA_COUNT))
        else:
            assert False

        print(__device)
        exp.device = __device

        count += 1

    return exp_list


# gonna work with ray?
def launch_experiments(exp_list, processes=4):
    set_start_method("spawn")
    with Pool(processes=processes) as pool:
        pool.map(start_experiment, exp_list)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=4, required=False)
    parser.add_argument('--device', type=str, default="cpu", required=False)
    parser.add_argument('--exp-path', type=str, default="generated/exp_list", required=False)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    launch_experiments(load_experiments(args.exp_path, args.device), args.processes)
