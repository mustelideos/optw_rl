#!/usr/bin/env python
# coding: utf-8

import math, operator
import numpy as np
import pandas as pd
import argparse

import os,time

from src.utils import get_instance_type, get_instance_df, get_distance_matrix
import src.config as cf
from src.sampling_norm_utils import sample_new_instance

import random
import torch

random.seed(2925)

#------------------------------------------------------------------------------------------

def setup_args_parser():
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--instance', help='which instance to run')
    parser.add_argument('--ni', help='number of generated instances', default=64, type=int)

    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')

    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    return parser


def parse_args_further(args):

    args.instance_type = get_instance_type(args.instance)
    args.output_directory = cf.GENERATED_INSTANCES_PATH+args.instance
    args.output_filename = 'inp_val_{sample_type}.pt'.format(sample_type=args.sample_type)
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    return args



if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    # for reproducibility"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_directory) if not os.path.exists(args.output_directory) else None

    df_inst = get_instance_df(args.instance, cf.BENCHMARK_INSTANCES_PATH, instance_type=args.instance_type)
    D = get_distance_matrix(df_inst, instance_type=args.instance_type)

    raw_data = df_inst[['x','y','duration','ti','tf','prof','Total Time']].values

    raw_data = torch.FloatTensor(raw_data).to(args.device)
    raw_distm =  torch.FloatTensor(D).to(args.device)

    inp_val = [sample_new_instance(raw_data, raw_distm, args) for x in range(args.ni)]

    output_path = '{output_dir}/{output_file}'.format(output_dir=args.output_directory,
                                                      output_file=args.output_filename)
    torch.save(inp_val, open(output_path, 'wb'))
