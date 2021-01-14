import torch
import random

from src.sampling_norm_utils import sample_new_instance, data_scaler
import src.problem_config as pcf


def reward_fn(data, sample_solution, device):
    """
    Returns:
        Tensor of shape [batch_size] containing rewards
    """

    batch_size = sample_solution[0].shape[0]
    tour_reward = torch.zeros(batch_size, device=device)

    for act_id in sample_solution:
        tour_reward += data[act_id, pcf.REWARD_IDX].squeeze(0)

    return tour_reward


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_step=5000):
    """Decay learning rate by a factor of 0.96 every lr_decay_epoch epochs.
       Lower_bounded at 0.00001"""
    lr = init_lr * (0.96**(epoch // lr_decay_step))
    if lr < 0.00001:
        lr = 0.00001

    if epoch % lr_decay_step == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def samples2batch(new_data, new_data_scaled, batch_size):
    bnew_data = new_data.expand(batch_size, -1, -1)
    bnew_data_scaled = new_data_scaled.expand(batch_size, -1, -1)
    return bnew_data, bnew_data_scaled


def train_model(raw_data, raw_dist_mat, norm_dic, run_episode, opt, args):

    new_data, start_time, dist_mat = sample_new_instance(raw_data, raw_dist_mat, args)
    new_data_scaled = data_scaler(new_data, norm_dic[args.instance])
    bnew_data, bnew_data_scaled = samples2batch(new_data, new_data_scaled, args.batch_size)

    run_episode.train()
    opt.zero_grad()
    actions, log_prob, entropy, step_mask = run_episode(bnew_data, bnew_data_scaled, start_time, dist_mat, 'stochastic')

    rewards = reward_fn(new_data, actions, args.device)

    loss = 0

    av_rew = rewards.mean()
    min_rew = rewards.min()
    max_rew = rewards.max()

    advantage = (rewards - av_rew) #advantage

    res = advantage.unsqueeze(1)*log_prob + args.beta*entropy

    loss = -res[step_mask].sum()/args.batch_size

    loss.backward(retain_graph=False)
    torch.nn.utils.clip_grad_norm_(run_episode.neuralnet.parameters(), args.max_grad_norm)
    opt.step()

    return av_rew.item(), min_rew.item(), max_rew.item(), loss.item()


def test_model(data, start_time, dist_mat, inst, inst_norm_dic, run_episode, device):
    with torch.no_grad():
        data_scaled = data_scaler(data, inst_norm_dic[inst])
        bdata, bdata_scaled = data.unsqueeze(0), data_scaled.unsqueeze(0)
        actions, log_prob, entropy, step_mask = run_episode(bdata, bdata_scaled, start_time, dist_mat, 'greedy')
        reward = reward_fn(data, actions, device)

        return reward.item()


def validation(inp_val, run_episode, inst_norm_dic, device):
    reward_val =  torch.tensor(0.0).to(device)
    rew_dict = {}
    for k, (inst_name, data) in enumerate(inp_val):
        inst_data, start_time, dist_mat = data
        rew = test_model(inst_data, start_time, dist_mat, inst_name, inst_norm_dic, run_episode, device)
        reward_val += rew
        key_str = inst_name + '_' + str(k)
        rew_dict[key_str] = rew

    return rew_dict, reward_val.item()/len(inp_val)
