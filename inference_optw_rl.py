import os,time
import logging
import argparse
import json
import pandas as pd
import numpy as np

import torch
from torch import optim

import src.inference_utils as iu
import src.utils as u
import src.sampling_norm_utils as snu

import src.config as cf
import src.problem_config as pcf

from src.neural_net import RecPointerNetwork


# for logging
N_DASHES = 40
SAVE_H_FILE = 'performance_scores.csv'




def setup_args_parser():
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--instance', help='which instance to run')
    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--use_checkpoint', help='use checkpoint (see https://pytorch.org/docs/stable/checkpoint.html)', action='store_true')
    parser.add_argument('--infe_type', help='which inference to run: \n \
                                greedy (gr), \
                                beam search (bs) or \
                                active search with beam search (as_bs)',
                                choices=['gr', 'bs', 'as_bs'],
                                default='bs')

    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')

    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--saved_model_epoch', help='epoch number which the pre-trained model was saved', default=500000, type=int)
    parser.add_argument('--model_name', help='model name', default='default', type=str)
    parser.add_argument('--nprint', help='epoch frequency for printing and saving in training history generated/benchmark instance reward', default=1, type=int)
    parser.add_argument('--nepocs', help='number of epochs for active search training', default=128, type=int)
    parser.add_argument('--batch_size', help='traing batch size', default = 32, type=int)
    parser.add_argument('--max_beam_number', help='-max number of beams in beam search inference', default = 128, type=int)
    parser.add_argument('--max_grad_norm', help='maximum norm value for gradient value clipping', default=1, type=int)
    parser.add_argument('--lr', help='learning rate for active search training', default=1e-5, type=float)
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    parser.add_argument('--beta', help='entropy term coefficient', default=0.01, type=float)
    parser.add_argument('--generated', help='run on the generated instances of the validation set\
                                             instead of on the benchmark instance', action='store_true')

    return parser




def parse_args_further(args):

    LOAD_W_DIR_STRING = '{results_path}/{benchmark_instance}/model_w/model_{model_name}_{sample_type}'
    GENERATED_STRING = '{generated_path}/{benchmark_instance}'

    VAL_SET_PT_FILE = 'inp_val_{sample_type}.pt'

    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    args.instance_type = u.get_instance_type(args.instance)
    args.map_location =  {'cpu': args.device_name}

    args.val_dir = GENERATED_STRING.format(generated_path=cf.GENERATED_INSTANCES_PATH,
                                           benchmark_instance=args.instance)

    args.load_w_dir = LOAD_W_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)

    args.val_set_pt_file = VAL_SET_PT_FILE.format(sample_type=args.sample_type)

    return args


def load_saved_args(args):

    with open(args.load_w_dir+'/model_'+args.model_name+'_training_args.txt') as json_file:
        data = json.load(json_file)
        args.n_layers = data['n_layers']
        args.n_heads = data['n_heads']
        args.ff_dim = data['ff_dim']
        args.nfeatures = data['nfeatures']
        args.ndfeatures = data['ndfeatures']
        args.rnn_hidden = data['rnn_hidden']

    return args


def log_args(args):
    logger.info(N_DASHES*'-')
    logger.info('Running test_optw_rl.py')
    logger.info(N_DASHES*'-')
    logger.info('model name:  %s' % args.model_name)
    logger.info(N_DASHES*'-')
    logger.info('device: %s' % args.device_name)
    logger.info('instance: %s' % args.instance)
    logger.info('instance type: %s' % args.instance_type)
    logger.info('infe_type: %s' % args.infe_type)
    logger.info('use_checkpoint: %s' % args.use_checkpoint)
    logger.info('sample type: %s' % args.sample_type)
    logger.info('sample_prof: %s' % args.sample_prof)
    logger.info('debug mode: %s' % args.debug)
    logger.info('nprint: %s' % args.nprint)
    logger.info('nepocs: %s' % args.nepocs)
    logger.info('seed: %s' % args.seed)
    logger.info(N_DASHES*'-')
    logger.info('max square length (Xmax): %s' % args.Xmax)
    logger.info('batch_size: %s' % args.batch_size)
    logger.info('max_grad_norm: %s' % args.max_grad_norm)
    logger.info('learning rate (lr): %s' % args.lr)
    logger.info('entropy term coefficient (beta): %s' % args.beta)
    logger.info('hidden size of RNN (hidden): %s' % args.rnn_hidden)
    logger.info('number of attention layers in the Encoder: %s' % args.n_layers)
    logger.info('number of features: %s' % args.nfeatures)
    logger.info('number of dynamic features: %s' % args.ndfeatures)
    logger.info(N_DASHES*'-')
    logger.info(args.instance)



if __name__ == "__main__":

    # ---------------------------------
    #  parse arguments and setup logger
    # ---------------------------------

    parser = setup_args_parser()
    args_temp = parser.parse_args()

    args = parse_args_further(args_temp)
    args = load_saved_args(args)

    logger = u.setup_logger(args.debug)
    if args.debug:
        log_args(args)


    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)


    # ---------------------------------
    #  load data
    # ---------------------------------
    inp_real = u.get_real_data(args, phase='inference')
    raw_data, raw_distm = inp_real[0]

    start_time = raw_data[0, pcf.OPENING_TIME_WINDOW_IDX]

    # get Tmax and Smax
    norm_dic = {}
    Tmax, Smax = snu.instance_dependent_norm_const(raw_data)
    norm_dic = {'Tmax': Tmax, 'Smax': Smax}

    # ---------------------------------
    # load model
    # ---------------------------------
    logger.info('Loading model for instance {instance} ...'.format(instance=args.instance))
    performance_scores = []

    pointer_net = RecPointerNetwork(args.nfeatures, args.ndfeatures,
                              args.rnn_hidden, args).to(args.device).eval()


    # ---------------------------------
    # inference
    # ---------------------------------

    if not args.generated:
        logger.info('Infering route for benchmark instance...')
        output =  iu.run_single(raw_data, norm_dic, start_time, raw_distm, args,
                                pointer_net, which_inf=args.infe_type)

    else:
        inp_val = u.get_val_data(args, phase='inference')
        logger.info('Infering routes for {num_inst} generated instances...' \
                    .format(num_inst=len(inp_val)))
        outputs =  iu.run_multiple(inp_val, norm_dic, args, pointer_net,
                                   which_inf=args.infe_type)

    # ---------------------------------
    # Log results
    # ---------------------------------
    if args.infe_type in ['gr', 'bs', 'as_bs']:

        logger.info(N_DASHES*'-')
        if not args.generated:
            logger.info('route: {route}'.format(route=output['route']))
            logger.info('total score: {total_score}'\
                        .format(total_score=int(output['score'])))
            inference_time_ms = int(1000*output['inf_time'])
            logger.info('inference time: {inference_time} ms'\
                        .format(inference_time=inference_time_ms))

        else:
            df_out = pd.DataFrame(outputs)
            average_total_score = round(df_out.score.mean(), 2)
            average_inf_time_ms = int(1000*df_out.inf_time.mean())
            logger.info('average total score: {average_total_score}' \
                        .format(average_total_score=average_total_score))
            logger.info('average inference time: {average_inference_time} ms' \
                        .format(average_inference_time=average_inf_time_ms))
        logger.info(N_DASHES*'-')
