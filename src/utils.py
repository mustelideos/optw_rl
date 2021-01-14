import pandas as pd
import numpy as np
import logging
import torch

import src.config as cf
import src.problem_config as pcf


def read_instance_data(instance_name, path):

    """reads instance data"""
    PATH_TO_BENCHMARK_INSTANCES = path

    benchmark_file = '{path_to_benchmark_instances}/{instance}.txt' \
                     .format(path_to_benchmark_instances=PATH_TO_BENCHMARK_INSTANCES,
                             instance=instance_name)

    dfile = open(benchmark_file)
    data = [[float(x) for x in line.split()] for line in dfile]
    dfile.close()
    return data


def eliminate_extra_cordeau_columns(instance_data):
    """Cordeau instances have extra columns in some rows. This function eliminates the extra columns.
    This will also correct position of total time in row 0 for all instances"""
    DATA_INIT_ROW = 2
    N_RELEVANT_FIRST_COLUMNS = 8
    N_RELEVANT_LAST_COLUMNS = 2

    return [s[:N_RELEVANT_FIRST_COLUMNS]+s[-N_RELEVANT_LAST_COLUMNS:] \
            for s in instance_data[DATA_INIT_ROW :]]


def test_n_vert_1(instance_data, instance_type):
    N_VERT_ROW = 0

    if instance_type=='Gavalas':
        N_VERT_COL = 3
        DATA_INIT_ROW = 1
    else:
        N_VERT_COL = 2
        DATA_INIT_ROW = 2

    n_vert = instance_data[N_VERT_ROW][N_VERT_COL]
    count_vert = len(instance_data)-(DATA_INIT_ROW+1)

    assert count_vert==n_vert, 'number of vertices doesnt match number of data rows'


def test_n_vert_2(instance_data, instance_type):
    N_VERT_ROW = 0
    if instance_type=='Gavalas':
        N_VERT_COL = 3
    else:
        N_VERT_COL = 2
    COLUMN_NAMES = ['vertex number', 'x coordinate', 'y coordinate',
                    'service duration or visiting time', 'profit of the location',
                    'not relevant 1', 'not relevant 2', 'not relevant 3',
                    'opening of time window', 'closing of time window']
    COLUMN_NAMES = [s.replace(' ', '_') for s in COLUMN_NAMES]

    VERTEX_NUMBER_COL = [i for i,n in enumerate(COLUMN_NAMES) if n=='vertex_number'][0]
    n_vert = instance_data[N_VERT_ROW][N_VERT_COL]
    last_vert_number = instance_data[-1][VERTEX_NUMBER_COL]

    assert last_vert_number==n_vert, 'number of vertices doesnt match vertice count of last row'


def test_n_vert_3(instance_data, instance_type):
    if instance_type=='Gavalas':
        N_DAYS_INDEX = 1
        n_days = int(np.array(instance_data[0])[N_DAYS_INDEX])
        assert n_days==1, 'not a single tour/1 day instance'
    else:
        pass


def parse_instance_data_Gavalas(instance_data):
    """parse instance data into dataframe"""

    # get start date
    N_DAYS_INDEX = 1
    START_DAY_INDEX = 2
    M = instance_data[0][N_DAYS_INDEX]
    SD =  int(instance_data[0][START_DAY_INDEX])

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 't',
                           'open_0', 'close_0', 'open_1', 'close_1',
                           'open_2', 'close_2', 'open_3', 'close_3',
                           'open_4', 'close_4', 'open_5', 'close_5',
                           'open_6', 'close_6', 'b']

    df = pd.DataFrame(instance_data[2:], columns=COLUMN_NAMES_ABBREV)

    df_ = df[['i', 'x', 'y', 'd', 'S', 't']+[c for c in df.columns if c[-1]==str(SD)]]
    columns = ['i', 'x', 'y', 'd', 'S', 't', OPENING_TIME_WINDOW_ABBREV_KEY, CLOSING_TIME_WINDOW_ABBREV_KEY]
    df_.columns=columns

    aux = pd.DataFrame([instance_data[1]], columns = ['i', 'x', 'y', 'd', 'S', 'O', 'C'])
    df = pd.concat([aux, df_], axis=0, sort=True).reset_index(drop=True)

    #add total time
    df[TOTAL_TIME_KEY] = 0
    df[TOTAL_TIME_KEY] = df.loc[0][CLOSING_TIME_WINDOW_ABBREV_KEY]

    return df


def parse_instance_data(instance_data):
    """parse instance data into dataframe"""

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 'f', 'a', 'list', 'O', 'C']

    instance_data_clean = eliminate_extra_cordeau_columns(instance_data)
    df = pd.DataFrame(instance_data_clean, columns=COLUMN_NAMES_ABBREV)

    #add total time
    df[TOTAL_TIME_KEY] = 0
    df[TOTAL_TIME_KEY] = df.loc[0][CLOSING_TIME_WINDOW_ABBREV_KEY]

    return df


def get_instance_df(instance_name, path, instance_type):

    """combine read instance, tests and parse to dataframe"""

    OPENING_TIME_WINDOW_ABBREV_KEY = 'O'
    CLOSING_TIME_WINDOW_ABBREV_KEY = 'C'
    TOTAL_TIME_KEY = 'Total Time'


    COLUMN_NAMES = ['vertex number', 'x coordinate', 'y coordinate',
    'service duration or visiting time', 'profit of the location',
    'not relevant 1', 'not relevant 2', 'not relevant 3',
    'opening of time window', 'closing of time window']
    COLUMN_NAMES = [s.replace(' ', '_') for s in COLUMN_NAMES]
    COLUMN_NAMES_ABBREV = ['i', 'x', 'y', 'd', 'S', 'f', 'a', 'list', 'O', 'C']
    VERTEX_NUMBER_COL = [i for i,n in enumerate(COLUMN_NAMES) if n=='vertex_number'][0]
    COLS_OF_INT = ['i', 'x', 'y', 'd', OPENING_TIME_WINDOW_ABBREV_KEY,
                   CLOSING_TIME_WINDOW_ABBREV_KEY, 'S', TOTAL_TIME_KEY]
    COLS_OF_INT_NEW_NAMES = ['i', 'x', 'y', 'duration', 'ti', 'tf', 'prof', TOTAL_TIME_KEY]

    standard2newnames_dict =  dict(((c, ca) for c, ca in zip(COLS_OF_INT, COLS_OF_INT_NEW_NAMES)))

    instance_data = read_instance_data(instance_name, path)

    # run tests
    test_n_vert_1(instance_data, instance_type)
    test_n_vert_2(instance_data, instance_type)
    # test if it's a single day (we are not considering TOPTW instances)
    test_n_vert_3(instance_data, instance_type)

    if instance_type=='Gavalas':
        df = parse_instance_data_Gavalas(instance_data)
    else:
        df = parse_instance_data(instance_data)

    #change column names
    COLS_OF_INT_NEW_NAMES = [standard2newnames_dict[s] for s in COLS_OF_INT]
    df_ = df[COLS_OF_INT].copy()
    df_.columns = COLS_OF_INT_NEW_NAMES
    df_['inst_name'] = instance_name
    df_['real_or_val'] = 'real'

    df_ = df_.append(df_.loc[0])
    return df_



def get_distance_matrix(instance_df, instance_type):
    """
    Distances between locations were rounded down to the first decimal
    for the Solomon instances and to the second decimal for the instances of Cordeau and Gavalas.
    """

    if instance_type in ['Solomon']:
        n_digits = 10.0

    elif instance_type in ['Cordeau', 'Gavalas']:
        n_digits = 100.0

    n = instance_df.shape[0]
    distm = np.zeros((n,n))
    x = instance_df.x.values
    y = instance_df.y.values

    for i in range(0, n-1):
        for j in range(i+1, n):
            distm[i,j] = np.floor(n_digits*(np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2)))/n_digits
            distm[j,i] = distm[i,j]

    return distm



def get_instance_type(instance_name):
        if instance_name[:2]=='pr':
            return 'Cordeau'
        elif instance_name[0] in ['r', 'c']:
            return 'Solomon'
        elif instance_name[0] in ['t']:
            return 'Gavalas'
        raise Exception('weird instance')


def setup_logger(debug):
    logger = logging.getLogger()
    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger


def get_instance_data(instance, path, device):
    instance_type = get_instance_type(instance)
    df_inst = get_instance_df(instance, path, instance_type)
    distm = get_distance_matrix(df_inst, instance_type)
    raw_data = df_inst[['x', 'y', 'duration', 'ti', 'tf', 'prof', 'Total Time']].values
    raw_data = torch.FloatTensor(raw_data).to(device)
    raw_distm =  torch.FloatTensor(distm).to(device)

    return raw_data, raw_distm


def get_val_data(args, phase='train'):
    path_string = '{directory}/{file_name}'
    inp_val_path = path_string.format(directory=args.val_dir, file_name=args.val_set_pt_file)
    inp_val = torch.load(inp_val_path, map_location = args.map_location)

    new_inp_val = [(args.instance, inst_data) for inst_data in inp_val]

    if phase=='train':
        return new_inp_val
    else:

        return inp_val


def get_real_data(args, phase='train'):

    df = get_instance_df(args.instance, cf.BENCHMARK_INSTANCES_PATH,
                         args.instance_type)
    dist_mat = get_distance_matrix(df, args.instance_type)
    inp_real = df[['x', 'y', 'duration', 'ti', 'tf', 'prof', 'Total Time']].values

    if phase=='train':
        inp_real = [(torch.FloatTensor(inp_real).to(args.device),
                torch.tensor(inp_real[0, pcf.OPENING_TIME_WINDOW_IDX]).to(args.device),
                torch.FloatTensor(dist_mat).to(args.device))]

        new_inp_real = [(args.instance, inp_real[0])]
        return new_inp_real
    else:
        inp_real = [(torch.FloatTensor(inp_real).to(args.device),
                 torch.FloatTensor(dist_mat).to(args.device))]
        return inp_real