import os
import logging
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tnrange, tqdm
import torch
from torch import optim

import src.config as cf
import src.utils as u
import src.train_utils as tu
import src.sampling_norm_utils as snu
from src.neural_net import RecPointerNetwork
from src.solution_construction import RunEpisode

# for logging
N_DASHES = 40


def train_loop(inp_val, inp_real, raw_data, run_episode, args):

    raw_data, raw_dist_mat = inp_real[0][1][0], inp_real[0][1][2]
    reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0
    training_history = []
    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.lr)
    step_dict = {}

    for epoch in tqdm(range(1,args.nepocs+1)):

        avg_reward, min_reward, max_reward, loss = tu.train_model(raw_data,
                                                                  raw_dist_mat,
                                                                  norm_dic,
                                                                  run_episode,
                                                                  model_opt,
                                                                  args)

        reward_total += avg_reward
        min_reward_total += min_reward
        max_reward_total += max_reward
        loss_total += loss

        tu.exp_lr_scheduler(model_opt, epoch, init_lr=args.lr)


        if epoch==0 or epoch % args.nprint == 0:
            logger.info("Epoch %s" % str(epoch))
            rew_dict, avg_reward_val = tu.validation(inp_val, run_episode, norm_dic, args.device)
            _, avg_reward_real  = tu.validation(inp_real, run_episode, norm_dic, args.device)
            step_dict[epoch] = rew_dict

            if epoch == 0:
                avg_loss = loss_total
                avg_reward_total = reward_total
                avg_min_reward_total = min_reward_total
                avg_max_reward_total = max_reward_total

                training_history.append([epoch, reward_total, min_reward_total, max_reward_total,
                                         avg_reward_val, avg_reward_real, loss_total])

            else:
                avg_loss = loss_total / args.nprint
                avg_reward_total = reward_total / args.nprint
                avg_min_reward_total = min_reward_total / args.nprint
                avg_max_reward_total = max_reward_total / args.nprint


                training_history.append([epoch, avg_reward_total, avg_min_reward_total, avg_max_reward_total,
                                         avg_reward_val, avg_reward_real, avg_loss])


            logger.info(N_DASHES*'-')
            logger.info("Average total loss: %s" % avg_loss)
            logger.info("Average train mean reward: %s" % avg_reward_total)
            logger.info("Average train max reward: %s" % avg_max_reward_total)
            logger.info("Average train min reward: %s" % avg_min_reward_total)
            logger.info("Validation reward: %2.3f"  % (avg_reward_val))
            logger.info("Real instance reward: %2.3f"  % (avg_reward_real))

            reward_total, min_reward_total, max_reward_total, loss_total = 0, 0, 0, 0

        if epoch % args.nsave == 0 and not args.debug:
            print('saving model')
            torch.save(run_episode.neuralnet.state_dict(), args.save_w_dir+'/model_'+str(epoch)+'.pkl')

    return training_history



def setup_args_parser():

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--instance', help='which instance to train on')
    parser.add_argument('--device', help='device to use (cpu/cuda)', default='cuda')
    parser.add_argument('--use_checkpoint', help='use checkpoint (see https://pytorch.org/docs/stable/checkpoint.html)', action='store_true')
    parser.add_argument('--sample_type', help='how to sample the scores of each point of interest: \n \
                                uniformly sampled (uni_samp), \
                                score proportional to each point of interest\'s duration of visit (corr_samp)' ,
                                choices=['uni_samp', 'corr_samp'],
                                default='uni_samp')
    parser.add_argument('--model_name', help='model name', default='default', type=str)
    parser.add_argument('--debug', help='debug mode (verbose output and no saving)', action='store_true')
    parser.add_argument('--nsave', help='saves the model weights every <nsave> epochs', default=10000, type=int)
    parser.add_argument('--nprint', help='to log and save the training history \
                                          (total score in the benchmark and generated \
                                          instances of the validation set) every <nprint> epochs', default=2500, type=int)
    parser.add_argument('--nepocs', help='number of training epochs', default=100000, type=int)
    parser.add_argument('--batch_size', help='training batch size', default=32, type=int)
    parser.add_argument('--max_grad_norm', help='maximum norm value for gradient value clipping', default=1, type=int)
    parser.add_argument('--lr', help='initial learning rate', default=1e-4, type=float)
    parser.add_argument('--seed', help='seed random # generators (for reproducibility)', default=2925, type=int)
    parser.add_argument('--beta', help='entropy term coefficient', default=0.01, type=float)
    parser.add_argument('--rnn_hidden', help='hidden size of RNN', default=128, type=int)
    parser.add_argument('--n_layers', help='number of attention layers in the encoder', default=2, type=int)
    parser.add_argument('--n_heads', help='number heads in attention layers', default=8, type=int)
    parser.add_argument('--ff_dim', help='hidden dimension of the encoder\'s feedforward sublayer', default=256, type=int)
    parser.add_argument('--nfeatures', help='number of non-dynamic features', default=7, type=int)
    parser.add_argument('--ndfeatures', help='number of dynamic features', default=8, type=int)

    return parser

def parse_args_further(args):

    SAVE_W_DIR_STRING = '{results_path}/{benchmark_instance}/model_w/model_{model_name}_{sample_type}'
    SAVE_H_DIR_STRING = '{results_path}/{benchmark_instance}/outputs/model_{model_name}_{sample_type}'
    GENERATED_DIR_STRING = '{generated_path}/{benchmark_instance}'

    args.save_h_file = 'training_history.csv'

    GENERATED_PT_FILE = 'inp_val_{sample_type}.pt'
    args.device_name = str(args.device)
    args.device = torch.device(args.device_name)
    args.instance_type = u.get_instance_type(args.instance)
    args.map_location =  {'cpu': args.device_name}
    args.save_w_dir = SAVE_W_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)
    args.save_h_dir = SAVE_H_DIR_STRING.format(results_path=cf.RESULTS_PATH,
                                               model_name=args.model_name,
                                               sample_type=args.sample_type,
                                               benchmark_instance = args.instance)

    args.val_dir = GENERATED_DIR_STRING.format(generated_path=cf.GENERATED_INSTANCES_PATH,
                                               benchmark_instance = args.instance)
    args.val_set_pt_file = GENERATED_PT_FILE.format(sample_type=args.sample_type)


    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    return args


def log_args(args):
    logger.info(N_DASHES*'-')
    logger.info('Running train_model_main.py')
    logger.info(N_DASHES*'-')
    logger.info('model name:  %s' % args.model_name)
    logger.info(N_DASHES*'-')
    logger.info('device: %s' % args.device_name)
    logger.info('instance: %s' % args.instance)
    logger.info('instance type: %s' % args.instance_type)
    logger.info('use_checkpoint: %s' % args.use_checkpoint)
    logger.info('sample type: %s' % args.sample_type)
    logger.info('debug mode: %s' % args.debug)
    logger.info('nsave: %s' % args.nsave)
    logger.info('nprint: %s' % args.nprint)
    logger.info('nepocs: %s' % args.nepocs)
    logger.info('seed: %s' % args.seed)
    logger.info(N_DASHES*'-')
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


def save_training_history(training_history, file_path, args):
    TRAIN_HIST_COLUMNS = ['epoch', 'avg_reward_train',
        'min_reward_train', 'max_reward_train', 'avg_reward_val_'+args.sample_type,
        'avg_reward_real', 'tloss_train']

    df = pd.DataFrame(training_history, columns = TRAIN_HIST_COLUMNS)
    df.to_csv(file_path, index=False)
    return


def save_args(args):
    keys_list = ['instance', 'n_layers', 'n_heads', 'ff_dim', 'nfeatures',
                 'ndfeatures', 'rnn_hidden', 'sample_type', 'lr', 'batch_size',
                 'seed', 'beta', 'max_grad_norm', 'nsave', 'nepocs']

    dict_to_save = { key: args.__dict__[key] for key in keys_list }
    FILE_NAME = '/model_'+args.model_name+'_training_args.txt'

    with open(args.save_w_dir+FILE_NAME, 'w') as f:
        json.dump(dict_to_save, f, indent=2)

    return


if __name__ == "__main__":

    # parse arguments and setup logger
    parser = setup_args_parser()
    args_temp = parser.parse_args()
    args = parse_args_further(args_temp)

    logger = u.setup_logger(args.debug)
    log_args(args)

    # for reproducibility"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if str(args.device) in ['cuda', 'cuda:0', 'cuda:1']:
        torch.cuda.manual_seed(args.seed)

    # create directories for saving
    if not args.debug:
        os.makedirs(args.save_h_dir) if not os.path.exists(args.save_h_dir) else None
        os.makedirs(args.save_w_dir) if not os.path.exists(args.save_w_dir) else None

    # get val and real instance scores
    inp_real = u.get_real_data(args, phase='train')
    inp_val = u.get_val_data(args, phase='train')

    # get Tmax and Smax
    raw_data = inp_real[0][1][0]
    Tmax, Smax = snu.instance_dependent_norm_const(raw_data)
    norm_dic = {args.instance: {'Tmax': Tmax, 'Smax': Smax}}

    # save args to file
    save_args(args)

    # train
    model = RecPointerNetwork(args.nfeatures, args.ndfeatures, args.rnn_hidden, args).to(args.device)
    run_episode = RunEpisode(model, args)

    training_history = train_loop(inp_val, inp_real, norm_dic, run_episode, args)

    # save
    if not args.debug:
        file_path = '%s/%s' % (args.save_h_dir, args.save_h_file)
        save_training_history(training_history, file_path, args)
