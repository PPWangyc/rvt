# read data.json

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config_data', type=str, default='./configs/data.json')
args = parser.parse_args()

with open(args.config_data) as f:
    data = json.load(f)

# get the keys of the dictionary
keys = data.keys()

# get the values of the dictionary
values = data.values()

train_set = data['train_set']
test_set = data['test_set']
select_test_set = data['select_test_set']

# insert test_set[0] into train_set[0]
train_set.insert(0, test_set[0])
test = train_set.pop(-1)
test_set = [test]
select_test_set = [test]

data['train_set'] = train_set
data['test_set'] = test_set
data['select_test_set'] = select_test_set

with open(args.config_data, 'w') as f:
    json.dump(data, f, indent=4)


