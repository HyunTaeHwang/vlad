#!/usr/bin/env python

import _init_paths
import os

devkit_path = 'data/ILSVRC2014/ILSVRC2014_devkit'
det_lists = os.path.join(devkit_path, "data/det_lists")
train_outfile = "train_100.txt"

def process_train_file(train_list, train_file):
    print "Processing train file {}".format(train_file)
    with open(os.path.join(det_lists,train_file), "r") as f:
        for line in f:
            line = line.strip('\n')
            train_list.append(line)

def process_train_files(train_files):
    train_list = []
    for train_file in train_files:
        process_train_file(train_list, train_file)
    return train_list

def main():
    train_txts = [f for f in os.listdir(det_lists) if os.path.isfile(os.path.join(det_lists, f))]
    train_files = []
    for train in train_txts:
        if "pos" in train:
            train_files.append(train)
    
    print train_files
    train_list =  process_train_files(train_files)
    print len(train_list)
    train_list = train_list[0:100]
    with open(os.path.join(det_lists,train_outfile), "w") as outfile:
        outfile.write("\n".join(train_list))
    

if __name__ == "__main__":
    main()