import torch

import src.config as cf
import src.problem_config as pcf


def sample_new_instance(inst_data, dist_mat, args):
    instance_type = args.instance_type
    sample_type  = args.sample_type

    if instance_type=='Solomon':
        n_digits = 10.0
        xy_inf = 0.
        xy_delta = 100.
    elif instance_type=='Cordeau':
        n_digits = 100.0
        xy_inf = -100.
        xy_delta = 200.
    elif instance_type=='Gavalas':
        n_digits = 100.0
        xy_inf = 0.
        xy_delta = 100.

    poit = inst_data.clone()
    n = inst_data.shape[0]

    prof = inst_data[1:-1, pcf.REWARD_IDX]
    durat_max = int(inst_data[1:-1, pcf.VIS_DURATION_TIME_IDX].max().item())

    day_duration = int(inst_data[:, pcf.CLOSING_TIME_WINDOW_IDX].max().item())

    t_init_real = int(inst_data[0, pcf.OPENING_TIME_WINDOW_IDX].item()) # starting time
    t_max_real = int(inst_data[0, pcf.ARRIVAL_TIME_IDX].item()) # max arrival time

    day_fract_inf = pcf.SAMP_DAY_FRAC_INF
    t_min = int(pcf.SAMP_DAY_FRAC_INF*day_duration)
    ub_t_init_val = pcf.UB_T_INIT_FRAC*day_duration
    lb_t_max_val = pcf.LB_T_MAX_FRAC*day_duration

    ub = int(torch.min(torch.tensor([ub_t_init_val,
                                     t_max_real+int(day_fract_inf*day_duration)])))
    t_init_val = torch.randint(int(t_init_real)-int(day_fract_inf*day_duration),
                               ub,
                               (1,))

    lb = int(torch.max(torch.tensor([lb_t_max_val, int(t_init_val)+t_min])))
    t_max_val = torch.randint(lb,
                              t_max_real+int(day_fract_inf*day_duration),
                              (1,))

    Smax = int(torch.round(pcf.MULTIPLE_SCORE*prof.max()).item())
    if sample_type == 'uni_samp':
        new_scores = torch.randint(1, Smax, (n,))
    elif sample_type == 'corr_samp':
        new_scores_unbound = (Smax/durat_max)*inst_data[:, pcf.VIS_DURATION_TIME_IDX] + pcf.CORR_SCORE_STD*torch.randn(n, device=args.device)
        new_scores = torch.round(torch.min(Smax*torch.ones(1, device=args.device),
                                           torch.max(torch.ones(n, device=args.device),
                                                     new_scores_unbound)))

    poit[:, pcf.REWARD_IDX] = new_scores

    #------------ correct first/last----------

    poit[0, pcf.REWARD_IDX] = 0 # starting point
    poit[n-1, pcf.REWARD_IDX] = 0 # ending point

    poit[0, pcf.X_COORDINATE_IDX] = float(xy_inf+xy_delta*torch.rand(1)) # starting point
    poit[n-1, pcf.X_COORDINATE_IDX] = poit[0, pcf.X_COORDINATE_IDX] # ending point

    poit[0, pcf.Y_COORDINATE_IDX] = float(xy_inf+xy_delta*torch.rand(1)) # starting point
    poit[n-1, pcf.Y_COORDINATE_IDX] = poit[0, pcf.Y_COORDINATE_IDX] # ending point

    poit[:, pcf.ARRIVAL_TIME_IDX] = t_max_val*torch.ones(n)
    poit[0, pcf.OPENING_TIME_WINDOW_IDX] = t_init_val*torch.ones(1)
    poit[n-1, pcf.OPENING_TIME_WINDOW_IDX] = t_init_val*torch.ones(1)
    poit[0, pcf.CLOSING_TIME_WINDOW_IDX] = t_max_val*torch.ones(1)
    poit[n-1, pcf.CLOSING_TIME_WINDOW_IDX] = t_max_val*torch.ones(1)

    start_time = t_init_val.clone().detach().to(args.device)
    dist_matn = dist_mat.clone()

    for j in range(1, n-1):
        dist_matn[0, j] = float(torch.floor(n_digits*(torch.sqrt((poit[0,0]-poit[j,0])**2+(poit[0,1]-poit[j,1])**2))).item()/n_digits)
        dist_matn[n-1, j] = dist_matn[0, j]

        dist_matn[j, 0] = dist_matn[0, j]
        dist_matn[j, n-1] = dist_matn[n-1, j]

    return poit, start_time, dist_matn


def data_scaler(data, norm_dic):
    datan = data.clone()
    datan[:, pcf.X_COORDINATE_IDX] /= pcf.X_MAX
    datan[:, pcf.Y_COORDINATE_IDX] /= pcf.X_MAX
    datan[:, pcf.VIS_DURATION_TIME_IDX] /= (datan[:, pcf.VIS_DURATION_TIME_IDX].max())
    datan[:, pcf.OPENING_TIME_WINDOW_IDX] /= norm_dic['Tmax']
    datan[:, pcf.CLOSING_TIME_WINDOW_IDX ] /= norm_dic['Tmax']
    datan[:, pcf.REWARD_IDX] /= norm_dic['Smax']
    datan[:, pcf.ARRIVAL_TIME_IDX] /= norm_dic['Tmax']

    return datan


def instance_dependent_norm_const(instance_raw_data):
    day_duration = int(instance_raw_data[:, pcf.CLOSING_TIME_WINDOW_IDX].max().item())
    t_max_real = int(instance_raw_data[0, pcf.ARRIVAL_TIME_IDX].item()) # max instance arrival time
    arrival_time_val_ub = t_max_real+int(pcf.SAMP_DAY_FRAC_INF*day_duration)
    Tmax = int(max(day_duration, arrival_time_val_ub)) # max possible time value
    Smax = int(torch.round(pcf.MULTIPLE_SCORE*instance_raw_data[1:-1, pcf.REWARD_IDX].max()).item()) # max score

    return Tmax, Smax
