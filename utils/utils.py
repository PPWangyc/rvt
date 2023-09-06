import json
import os
import torch
import numpy as np
import random

# def get current test set

def get_current_test_set(data):
    with open(data) as f:
        data = json.load(f)
    return data['test_set']

# def output accuracy to json file

def output_data(output_dict):
    subject, accuracy, size, precision, recall, f1, path, bal_acc, y_true, y_pred = output_dict['subject'], output_dict['accuracy'], output_dict['size'], output_dict['precision'], output_dict['recall'], output_dict['f1'], output_dict['path'], output_dict['bal_acc'], output_dict['y_true'], output_dict['y_pred']
    with open(os.path.join(path), 'r') as f:
        data = json.load(f)
    # add data to the json file
    data[subject + '_acc'] = accuracy
    data[subject + '_size'] = size

    data[subject + '_precision'] = precision
    data[subject + '_recall'] = recall
    data[subject + '_f1'] = f1
    data[subject + '_ba'] = bal_acc
    data[subject + '_y_true'] = y_true
    data[subject + '_y_pred'] = y_pred
    # write the json file
    with open(os.path.join(path), 'w') as f:
        json.dump(data, f)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print('seed set to {}'.format(seed))
