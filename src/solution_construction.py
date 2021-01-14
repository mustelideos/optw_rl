import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.distributions import Categorical

import sys

from src.features_utils import DynamicFeatures
import src.problem_config as pcf

ourlogzero = sys.float_info.min


class Lookahead():
    def __init__(self, args):
        super(Lookahead, self).__init__()

        self.device = args.device
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.vis_duration_time_idx = pcf.VIS_DURATION_TIME_IDX
        self.arrival_time_idx = pcf.ARRIVAL_TIME_IDX

    def adjacency_matrix(self, braw_inputs, mask, dist_mat, pres_act, present_time):
        # feasible neighborhood for each node
        maskk = mask.clone()
        step_batch_size, npoints = mask.shape

        #one step forward update
        arrivej = dist_mat[pres_act] + present_time
        farrivej = arrivej.view(step_batch_size, npoints)
        tw_start = braw_inputs[:, :, self.opening_time_window_idx]
        waitj = torch.max(torch.FloatTensor([0.0]).to(self.device), tw_start-farrivej)
        durat = braw_inputs[:, : , self.vis_duration_time_idx]

        fpresent_time = farrivej + waitj + durat
        fpres_act = torch.arange(0, npoints, device=self.device).expand(step_batch_size, -1)

        # feasible neighborhood for each node
        adj_mask = maskk.unsqueeze(1).repeat(1, npoints, 1)
        arrivej = dist_mat.expand(step_batch_size, -1, -1) + fpresent_time.unsqueeze(2)
        waitj = torch.max(torch.FloatTensor([0.0]).to(self.device), tw_start.unsqueeze(2)-arrivej)

        tw_end = braw_inputs[:, :, self.closing_time_window_idx]
        ttime = braw_inputs[:, 0, self.arrival_time_idx]

        dlast = dist_mat[:, -1].unsqueeze(0).expand(step_batch_size, -1)

        c1 = arrivej + waitj <= tw_end.unsqueeze(1)
        c2 = arrivej + waitj + durat.unsqueeze(1) + dlast.unsqueeze(1) <= ttime.unsqueeze(1).unsqueeze(1).expand(-1, npoints, npoints)
        adj_mask = adj_mask * c1 * c2

        # self-loop
        idx = torch.arange(0, npoints, device=self.device).expand(step_batch_size, -1)
        adj_mask[:, idx, idx] = 1

        return adj_mask



class ModelUtils():
    def __init__(self, args):
        super(ModelUtils, self).__init__()

        self.device = args.device
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.vis_duration_time_idx = pcf.VIS_DURATION_TIME_IDX
        self.arrival_time_idx = pcf.ARRIVAL_TIME_IDX

    def feasibility_control(self, braw_inputs, mask, dist_mat, pres_act, present_time, batch_idx, first_step=False):

        done = False
        maskk = mask.clone()
        step_batch_size = batch_idx.shape[0]

        arrivej = dist_mat[pres_act] + present_time
        waitj = torch.max(torch.FloatTensor([0.0]).to(self.device), braw_inputs[:, :, self.opening_time_window_idx]-arrivej)

        c1 = arrivej + waitj <= braw_inputs[:, :, self.closing_time_window_idx]
        c2 = arrivej + waitj + braw_inputs[:, :, self.vis_duration_time_idx] + dist_mat[:, -1] <= braw_inputs[0, 0, self.arrival_time_idx]

        if not first_step:
            maskk[batch_idx, pres_act] = 0

        maskk[batch_idx] = maskk[batch_idx] * c1 * c2

        if maskk[:, -1].any() == 0:
            done = True
        return done, maskk


    def one_step_update(self, raw_inputs_b, dist_mat, pres_action, future_action, present_time, batch_idx, batch_size):

        present_time_b = torch.zeros(batch_size, 1, device=self.device)
        pres_actions_b = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        step_mask_b = torch.zeros(batch_size, 1, device=self.device, requires_grad=False, dtype=torch.bool)

        arrive_j = dist_mat[pres_action, future_action].unsqueeze(1) + present_time
        wait_j = torch.max(torch.FloatTensor([0.0]).to(self.device),
                           raw_inputs_b[batch_idx, future_action, self.opening_time_window_idx].unsqueeze(1)-arrive_j)
        present_time = arrive_j + wait_j + raw_inputs_b[batch_idx, future_action, self.vis_duration_time_idx].unsqueeze(1)

        present_time_b[batch_idx] = present_time

        pres_actions_b[batch_idx] = future_action
        step_mask_b[batch_idx] = 1

        return pres_actions_b, present_time_b, step_mask_b



class RunEpisode(nn.Module):

    def __init__(self, neuralnet, args):
        super(RunEpisode, self).__init__()

        self.device = args.device
        self.neuralnet = neuralnet
        self.dyn_feat = DynamicFeatures(args)
        self.lookahead = Lookahead(args)
        self.mu = ModelUtils(args)

    def forward(self, binputs, bdata_scaled, start_time, dist_mat, infer_type):

        self.batch_size, sequence_size, input_size = binputs.size()

        h_0, c_0 = self.neuralnet.decoder.hidden_0

        dec_hidden = (h_0.expand(self.batch_size, -1), c_0.expand(self.batch_size, -1))

        mask = torch.ones(self.batch_size, sequence_size, device=self.device, requires_grad=False, dtype = torch.uint8)

        bpresent_time = start_time*torch.ones(self.batch_size, 1, device=self.device)

        llog_probs, lactions, lstep_mask, lentropy = [], [], [], []

        bpres_actions = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)

        batch_idx = torch.arange(0, self.batch_size, device=self.device)

        done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat, bpres_actions,
                                                 bpresent_time, batch_idx, first_step=True)

        adj_mask = self.lookahead.adjacency_matrix(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time)

        # encoder first forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time, bpres_actions, dist_mat, batch_idx)
        emb1 = self.neuralnet.sta_emb(bdata_scaled)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1,emb2), dim=2)

        _, _, enc_outputs = self.neuralnet(enc_inputs, enc_inputs, adj_mask, enc_inputs, dec_hidden, mask, first_step=True)

        decoder_input = enc_outputs[batch_idx, bpres_actions]

        done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time, batch_idx)
        adj_mask = self.lookahead.adjacency_matrix(binputs[batch_idx], mask, dist_mat, bpres_actions, bpresent_time)

        # encoder/decoder forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time,
                                                      bpres_actions, dist_mat, batch_idx)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1,emb2), dim=2)

        policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

        lactions.append(bpres_actions)

        # Starting the trip
        while not done:

            future_actions, log_probs, entropy = self.select_actions(policy, infer_type)

            bpres_actions, bpresent_time, bstep_mask = self.mu.one_step_update(binputs, dist_mat, bpres_actions[batch_idx],
                                                                               future_actions, bpresent_time[batch_idx],
                                                                               batch_idx, self.batch_size)

            blog_probs = torch.zeros(self.batch_size, 1, dtype=torch.float32).to(self.device)
            blog_probs[batch_idx] = log_probs.unsqueeze(1)

            bentropy = torch.zeros(self.batch_size,1,dtype=torch.float32).to(self.device)
            bentropy[batch_idx] = entropy.unsqueeze(1)

            llog_probs.append(blog_probs)
            lactions.append(bpres_actions)
            lstep_mask.append(bstep_mask)
            lentropy.append(bentropy)

            done, mask = self.mu.feasibility_control(binputs[batch_idx], mask, dist_mat,
                                                     bpres_actions[batch_idx], bpresent_time[batch_idx],
                                                     batch_idx)

            if done: break
            sub_batch_idx = torch.nonzero(mask[batch_idx][:,-1], as_tuple=False).squeeze(1)

            batch_idx = torch.nonzero(mask[:,-1], as_tuple=False).squeeze(1)

            adj_mask = self.lookahead.adjacency_matrix(binputs[batch_idx], mask[batch_idx], dist_mat, bpres_actions[batch_idx], bpresent_time[batch_idx])

            #update decoder input and hidden
            decoder_input = enc_outputs[sub_batch_idx, bpres_actions[sub_batch_idx]]
            dec_hidden = (dec_hidden[0][sub_batch_idx], dec_hidden[1][sub_batch_idx])

            # encoder/decoder forward pass
            bdyn_inputs = self.dyn_feat.make_dynamic_feat(binputs, bpresent_time[batch_idx], bpres_actions[batch_idx], dist_mat, batch_idx)
            emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
            enc_inputs = torch.cat((emb1[batch_idx],emb2), dim=2)

            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs[sub_batch_idx], adj_mask, decoder_input, dec_hidden, mask[batch_idx])

        return lactions, torch.cat(llog_probs, dim=1), torch.cat(lentropy, dim=1), torch.cat(lstep_mask, dim=1)


    def select_actions(self, policy, infer_type):

        if infer_type == 'stochastic':
            m = Categorical(policy)
            act_ind = m.sample()
            log_select =  m.log_prob(act_ind)
            poli_entro = m.entropy()
        elif infer_type == 'greedy':
            prob, act_ind = torch.max(policy, 1)
            log_select =  prob.log()
            poli_entro =  torch.zeros(self.batch_size, requires_grad=False).to(self.device)

        return act_ind, log_select, poli_entro



class BeamSearch(nn.Module):
    def __init__(self, neuralnet, args):
        super(BeamSearch, self).__init__()

        self.device = args.device
        self.neuralnet = neuralnet
        self.dyn_feat = DynamicFeatures(args)
        self.lookahead = Lookahead(args)
        self.mu = ModelUtils(args)

    def forward(self, inputs, data_scaled, start_time, dist_mat, infer_type, beam_size):
        self.beam_size = beam_size
        _, sequence_size, input_size = inputs.size()

        # first step  - node 0
        bpresent_time = start_time*torch.ones(1, 1, device=self.device)

        mask = torch.ones(1, sequence_size, device=self.device, requires_grad=False, dtype= torch.uint8)
        bpres_actions = torch.zeros(1, dtype=torch.int64,device=self.device)
        beam_idx = torch.arange(0, 1, device=self.device)

        done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                 mask, dist_mat, bpres_actions, bpresent_time,
                                                 torch.arange(0, mask.shape[0], device=self.device),
                                                 first_step=True)
        adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                   mask, dist_mat, bpres_actions, bpresent_time)

        h_0, c_0 = self.neuralnet.decoder.hidden_0
        dec_hidden = (h_0.expand(1, -1), c_0.expand(1, -1))

        step = 0

        # encoder first forward pass
        bdata_scaled = data_scaled.expand(1,-1,-1)
        sum_log_probs = torch.zeros(1, device=self.device).float()

        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(1,-1,-1), bpresent_time, bpres_actions, dist_mat, beam_idx)
        emb1 = self.neuralnet.sta_emb(bdata_scaled)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1, emb2), dim=2)

        _, _, enc_outputs = self.neuralnet(enc_inputs, enc_inputs, adj_mask, enc_inputs, dec_hidden, mask, first_step=True)

        decoder_input = enc_outputs[beam_idx, bpres_actions]

        done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                 mask, dist_mat, bpres_actions, bpresent_time,
                                                 torch.arange(0, mask.shape[0], device=self.device))
        adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                   mask, dist_mat, bpres_actions, bpresent_time)

        # encoder/decoder forward pass
        bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(beam_idx.shape[0], -1, -1), bpresent_time, bpres_actions, dist_mat, beam_idx)
        emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
        enc_inputs = torch.cat((emb1,emb2), dim=2)

        policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

        future_actions, log_probs, beam_idx = self.select_actions(policy, sum_log_probs, mask, infer_type)
        # info update
        h_step = torch.index_select(dec_hidden[0], dim=0, index = beam_idx)
        c_step = torch.index_select(dec_hidden[1], dim=0, index = beam_idx)
        dec_hidden = (h_step,c_step)

        mask = torch.index_select(mask, dim=0, index=beam_idx)
        bpresent_time = torch.index_select(bpresent_time, dim=0, index=beam_idx)
        bpres_actions = torch.index_select(bpres_actions, dim=0, index=beam_idx)
        enc_outputs  = torch.index_select(enc_outputs, dim=0, index=beam_idx)
        sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=beam_idx)

        emb1 = torch.index_select(emb1, dim=0, index=beam_idx)

        # initialize buffers
        bllog_probs = torch.zeros(bpres_actions.shape[0], sequence_size, device=self.device).float()
        blactions = torch.zeros(bpres_actions.shape[0], sequence_size, device=self.device).long()

        sum_log_probs += log_probs.squeeze(0).detach()

        blactions[:, step] = bpres_actions

        final_log_probs, final_actions, lstep_mask = [], [], []

        # Starting the trip
        while not done:

            future_actions = future_actions.squeeze(0)

            beam_size = bpres_actions.shape[0]
            bpres_actions, bpresent_time, bstep_mask = \
                self.mu.one_step_update(inputs.expand(beam_size, -1, -1), dist_mat,
                                        bpres_actions, future_actions, bpresent_time,
                                        torch.arange(0,beam_size,device=self.device),
                                        beam_size)

            bllog_probs[:, step] = log_probs
            blactions[:, step+1] = bpres_actions
            step+=1

            done, mask = self.mu.feasibility_control(inputs.expand(beam_idx.shape[0], -1, -1),
                                                     mask, dist_mat, bpres_actions, bpresent_time,
                                                     torch.arange(0, mask.shape[0], device=self.device))
            adj_mask = self.lookahead.adjacency_matrix(inputs.expand(beam_idx.shape[0], -1, -1),
                                                       mask, dist_mat, bpres_actions, bpresent_time)

            active_beam_idx = torch.nonzero(mask[:, -1], as_tuple=False).squeeze(1)
            end_beam_idx = torch.nonzero((mask[:, -1]==0), as_tuple=False).squeeze(1)

            if end_beam_idx.shape[0]>0:

                final_log_probs.append(torch.index_select(bllog_probs, dim=0, index=end_beam_idx))
                final_actions.append(torch.index_select(blactions, dim=0, index=end_beam_idx))

                # ending seq info update
                h_step = torch.index_select(dec_hidden[0], dim=0, index = active_beam_idx)
                c_step = torch.index_select(dec_hidden[1], dim=0, index = active_beam_idx)
                dec_hidden = (h_step,c_step)

                mask = torch.index_select(mask, dim=0, index=active_beam_idx)
                adj_mask = torch.index_select(adj_mask, dim=0, index=active_beam_idx)

                bpresent_time = torch.index_select(bpresent_time, dim=0, index=active_beam_idx)
                bpres_actions = torch.index_select(bpres_actions, dim=0, index=active_beam_idx)
                enc_outputs  = torch.index_select(enc_outputs, dim=0, index=active_beam_idx)

                emb1 = torch.index_select(emb1, dim=0, index=active_beam_idx)

                blactions = torch.index_select(blactions, dim=0, index=active_beam_idx)
                bllog_probs = torch.index_select(bllog_probs, dim=0, index=active_beam_idx)
                sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=active_beam_idx)

            if done: break
            decoder_input = enc_outputs[torch.arange(0, bpres_actions.shape[0], device=self.device), bpres_actions]

            bdyn_inputs = self.dyn_feat.make_dynamic_feat(inputs.expand(beam_idx.shape[0], -1, -1), bpresent_time, bpres_actions, dist_mat, active_beam_idx)
            emb2 = self.neuralnet.dyn_emb(bdyn_inputs)
            enc_inputs = torch.cat((emb1,emb2), dim=2)

            policy, dec_hidden, enc_outputs = self.neuralnet(enc_inputs, enc_outputs, adj_mask, decoder_input, dec_hidden, mask)

            future_actions, log_probs, beam_idx = self.select_actions(policy, sum_log_probs, mask, infer_type)

            # info update
            h_step = torch.index_select(dec_hidden[0], dim=0, index = beam_idx)
            c_step = torch.index_select(dec_hidden[1], dim=0, index = beam_idx)
            dec_hidden = (h_step,c_step)

            mask = torch.index_select(mask, dim=0, index=beam_idx)
            adj_mask = torch.index_select(adj_mask, dim=0, index=beam_idx)

            bpresent_time = torch.index_select(bpresent_time, dim=0, index=beam_idx)
            bpres_actions = torch.index_select(bpres_actions, dim=0, index=beam_idx)

            enc_outputs  = torch.index_select(enc_outputs, dim=0, index=beam_idx)

            emb1 = torch.index_select(emb1, dim=0, index=beam_idx)

            blactions = torch.index_select(blactions, dim=0, index=beam_idx)
            bllog_probs = torch.index_select(bllog_probs, dim=0, index=beam_idx)
            sum_log_probs = torch.index_select(sum_log_probs, dim=0, index=beam_idx)

            sum_log_probs += log_probs.squeeze(0).detach()

        return torch.cat(final_actions, dim=0), torch.cat(final_log_probs, dim=0)



    def select_actions(self, policy, sum_log_probs, mask, infer_type = 'stochastic'):

        beam_size, seq_size = policy.size()
        nzn  = torch.nonzero(mask, as_tuple=False).shape[0]
        sample_size = min(nzn,self.beam_size)

        ourlogzero = sys.float_info.min
        lpolicy = policy.masked_fill(mask == 0, ourlogzero).log()
        npolicy = sum_log_probs.unsqueeze(1) + lpolicy
        if infer_type == 'stochastic':
            nnpolicy = npolicy.exp().masked_fill(mask == 0, 0).view(1, -1)

            m = Categorical(nnpolicy)
            gact_ind = torch.multinomial(nnpolicy, sample_size)
            log_select =  m.log_prob(gact_ind)

        elif infer_type == 'greedy':
            nnpolicy = npolicy.exp().masked_fill(mask == 0, 0).view(1, -1)

            _ , gact_ind = nnpolicy.topk(sample_size, dim = 1)
            prob = policy.view(-1)[gact_ind]
            log_select =  prob.log()

        beam_id = torch.floor_divide(gact_ind, seq_size).squeeze(0)
        act_ind = torch.fmod(gact_ind, seq_size)

        return act_ind, log_select, beam_id
