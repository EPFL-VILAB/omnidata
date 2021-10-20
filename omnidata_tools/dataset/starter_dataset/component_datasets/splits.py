import csv
import os

def get_splits(split_path, forbidden_buildings=[]):
    with open(split_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        train_list, val_list, test_list = [], [], []
        for row in readCSV:
            name, is_train, is_val, is_test = row
            if name in forbidden_buildings: continue
            if is_train == '1': train_list.append(name)
            if is_val == '1':   val_list.append(name)
            if is_test == '1':  test_list.append(name)
    return {
        'train': sorted(train_list),
        'val': sorted(val_list),
        'test': sorted(test_list)
    }

def get_all_spaces(splits): return set(sum(splits.values(), start=[]))