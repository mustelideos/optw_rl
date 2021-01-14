from tqdm import tnrange, tqdm
import time


import torch
from torch import optim
import src.train_utils as tu
import src.sampling_norm_utils as snu
from src.solution_construction import RunEpisode, BeamSearch

def gr_inference(inst_data, norm_dic, start_time, dist_mat, args, run_episode):

    data_scaled = snu.data_scaler(inst_data, norm_dic)
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    with torch.no_grad():
        seq, _ , _, _ = run_episode(binst_data, bdata_scaled, start_time, dist_mat, 'greedy')

    rewards = tu.reward_fn(inst_data, seq, args.device)
    maxrew, idx_max = torch.max(rewards, 0)
    score = maxrew.item()

    route =  [act.item() for act in seq]
    route[-1] = 0
    return route, score


def bs_inference(inst_data, norm_dic, start_time, dist_mat, args, run_episode):

    data_scaled = snu.data_scaler(inst_data, norm_dic)
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    nb = args.max_beam_number
    with torch.no_grad():
        seq, _ = run_episode(binst_data, bdata_scaled, start_time, dist_mat, 'greedy', nb)

    seq_list = [ seq[:,k] for k in range(seq.shape[1])]
    rewards = tu.reward_fn(inst_data, seq_list, args.device)
    maxrew, idx_max = torch.max(rewards, 0)
    score = maxrew.item()

    route =  [0] + [val.item() for val in seq[idx_max] if val.item() != 0]
    route[-1] = 0
    return route, score


def as_bs_inference(inp_data, norm_dic, args, run_episode, run_episode_bs):

    model_opt = optim.Adam(run_episode.neuralnet.parameters(), lr=args.lr)

    inst_data, start_time, dist_mat = inp_data

    data_scaled = snu.data_scaler(inst_data, norm_dic)

    for epoch in tqdm(range(args.nepocs)):

        active_search_train_model(inst_data, data_scaled, start_time, dist_mat, run_episode, model_opt, args)

    # .. to load your previously training model:
    run_episode_bs.neuralnet.load_state_dict(run_episode.neuralnet.state_dict())
    binst_data, bdata_scaled = inst_data.unsqueeze(0), data_scaled.unsqueeze(0)

    with torch.no_grad():  
        seq, _ = run_episode_bs(binst_data, bdata_scaled, start_time, dist_mat, 'greedy', args.max_beam_number)

    seq_list = [ seq[:,k] for k in range(seq.shape[1])]
    rewards = tu.reward_fn(inst_data, seq_list, args.device)

    maxreward, idx_max = torch.max(rewards, 0)

    score = maxreward.item()

    route =  [0]+[val.item() for val in seq[idx_max] if val.item() != 0]
    route[-1] = 0

    return route, score

def run_single(inst_data, norm_dic, start_time, dist_mat, args, model,
               which_inf=None):


    saved_model_path = args.load_w_dir +'/model_' + str(args.saved_model_epoch) + '.pkl'
    model._load_model_weights(saved_model_path, args.device)


    tic = time.time()
    if which_inf=='bs':
        run_episode_inf = BeamSearch(model, args).eval()
        route, score = bs_inference(inst_data, norm_dic, start_time, dist_mat,
                                    args, run_episode_inf)

    elif which_inf=='gr':
        run_episode_inf = RunEpisode(model, args).eval()
        route, score = gr_inference(inst_data, norm_dic, start_time, dist_mat,
                                    args, run_episode_inf)

    elif which_inf=='as_bs':

        saved_model_path = args.load_w_dir +'/model_' + str(args.saved_model_epoch) + '.pkl'
        model._load_model_weights(saved_model_path, args.device)
        run_episode_train = RunEpisode(model, args)

        run_episode_inf = BeamSearch(model, args).eval()
        inp_data = (inst_data, start_time, dist_mat)
        route, score = as_bs_inference(inp_data, norm_dic, args,
                                       run_episode_train, run_episode_inf)
    toc = time.time()

    output = dict([('score', score), ('route', route), ('inf_time', toc-tic)])

    return output

def run_multiple(inp_val, norm_dic, args, model, which_inf=None):

    outputs = list()
    for k, (inst_data, start_time, dist_mat) in enumerate(tqdm(inp_val)):
        output = run_single(inst_data, norm_dic, start_time, dist_mat, args,
                               model, which_inf=which_inf)
        outputs.append(output)

    return outputs

def active_search_train_model(inst_data, data_scaled, inp_t_init_val, dist_mat, run_episode, model_opt, args):

    run_episode.train()

    binst_data, bdata_scaled = tu.samples2batch(inst_data, data_scaled, args.batch_size)

    actions, log_prob, entropy, step_mask = run_episode(binst_data, bdata_scaled, inp_t_init_val, dist_mat, 'stochastic')

    rewards = tu.reward_fn(inst_data, actions, args.device)

    av_rew = rewards.mean()

    advantage = (rewards - av_rew)

    res = advantage.unsqueeze(1)*log_prob + args.beta*entropy

    loss = -res[step_mask].sum()/args.batch_size

    model_opt.zero_grad()
    loss.backward(retain_graph=False)
    torch.nn.utils.clip_grad_norm_(run_episode.neuralnet.parameters(), args.max_grad_norm)
    model_opt.step()
