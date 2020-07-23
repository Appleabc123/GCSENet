import csv
import torch as t
import random


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def prepare_data(opt):
    dataset = dict()
    # dataset['gd_p'] = read_csv(opt.data_path + '/g-d.csv')      #gene-disease
    # dataset['gd_true'] = read_csv(opt.data_path + '/g-d.csv')
    dataset['gm_p'] = read_csv(opt.data_path + '\\g-m.csv')       #gene-miRNA
    dataset['gm_true'] = read_csv(opt.data_path + '\\g-m.csv')

    zero_index = []
    one_index = []
    # for i in range(dataset['gd_p'].size(0)):
    #     for j in range(dataset['gd_p'].size(1)):
    #         if dataset['gd_p'][i][j] < 1:
    #             zero_index.append([i, j])
    #         if dataset['gd_p'][i][j] >= 1:
    #             one_index.append([i, j])
    for i in range(dataset['gm_p'].size(0)):
        for j in range(dataset['gm_p'].size(1)):
            if dataset['gm_p'][i][j] < 1:
                zero_index.append([i, j])
            if dataset['gm_p'][i][j] >= 1:
                one_index.append([i, j])
    random.shuffle(one_index)
    random.shuffle(zero_index)
    zero_tensor = t.LongTensor(zero_index)
    one_tensor = t.LongTensor(one_index)
    # dataset['gd'] = dict()
    # dataset['gd']['train'] = [one_tensor, zero_tensor]
    dataset['gm'] = dict()
    dataset['gm']['train'] = [one_tensor, zero_tensor]

    # dd_matrix = read_csv(opt.data_path + '\\d-d.csv')
    # dd_edge_index = get_edge_index(dd_matrix)
    # dataset['dd'] = {'data': dd_matrix, 'edge_index': dd_edge_index}

    mm_matrix = read_csv(opt.data_path + '\\m-m.csv')
    mm_edge_index = get_edge_index(mm_matrix)
    dataset['mm'] = {'data': mm_matrix, 'edge_index': mm_edge_index}

    gg_matrix = read_csv(opt.data_path + '\\g-g.csv')
    gg_edge_index = get_edge_index(gg_matrix)
    dataset['gg'] = {'data': gg_matrix, 'edge_index': gg_edge_index}
    return dataset

