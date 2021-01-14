import torch
import src.problem_config as pcf



class DynamicFeatures():

    def __init__(self, args):
        super(DynamicFeatures, self).__init__()

        self.arrival_time_idx = pcf.ARRIVAL_TIME_IDX
        self.opening_time_window_idx = pcf.OPENING_TIME_WINDOW_IDX
        self.closing_time_window_idx = pcf.CLOSING_TIME_WINDOW_IDX
        self.device = args.device

    def make_dynamic_feat(self, data, current_time, current_poi_idx, dist_mat, batch_idx):

        num_dyn_feat = 8
        _ , sequence_size, input_size  = data.size()
        batch_size = batch_idx.shape[0]

        dyn_feat = torch.ones(batch_size, sequence_size, num_dyn_feat).to(self.device)

        tour_start_time = data[0, 0, self.opening_time_window_idx]
        max_tour_duration = data[0, 0, self.arrival_time_idx] - tour_start_time
        arrive_j_times = current_time + dist_mat[current_poi_idx]

        dyn_feat[:, :, 0] = (data[batch_idx, :, self.opening_time_window_idx] - current_time) / max_tour_duration
        dyn_feat[:, :, 1] = (data[batch_idx, :, self.closing_time_window_idx] - current_time) / max_tour_duration
        dyn_feat[:, :, 2] = (data[batch_idx, :, self.arrival_time_idx] - current_time) / max_tour_duration
        dyn_feat[:, :, 3] = (current_time - tour_start_time) / max_tour_duration


        dyn_feat[:, :, 4] = (arrive_j_times - tour_start_time) / max_tour_duration
        dyn_feat[:, :, 5] = (data[batch_idx, :, self.opening_time_window_idx] - arrive_j_times) / max_tour_duration
        dyn_feat[:, :, 6] = (data[batch_idx, :, self.closing_time_window_idx] - arrive_j_times) / max_tour_duration
        dyn_feat[:, :, 7] = (data[batch_idx, :, self.arrival_time_idx] - arrive_j_times) / max_tour_duration

        return dyn_feat
