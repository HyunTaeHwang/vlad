#!/usr/bin/env python

import _init_paths
import os
import xml.dom.minidom as minidom
from PIL import Image
# 
# Question 2: training dataset object aspect ratio has certain requirements. Otherwise, in RPN training iteration reported the following error because the aspect ratio is too large or too small, the resulting anchors not find a suitable anchor during the election inside_anchors plug in such overlaps only 0 elements.
# 
# anchor_target_layer.py ", Line 137, in Forward  
# gt_argmax_overlaps = overlaps.argmax (Axis = 0)  
# ValueError: attempt to GET AN argmax of empty Sequence
# object's bounding box aspect ratio 
# VOC2007: between 0.117-15.500
# ImageNet (ILSVRC2014): between 0.03-48.50
# The aspect ratio of the data constraints, 
# at least in order to ensure training 
# 0.117-15.500
# Question 3: training data set image width and height can not be too small. Otherwise, do im_proposal when suspended animation phenomenon occurs.

# Min, max aspect ratio 0.0645161290323 15.0322580645


min_ratio, max_ratio = 0.07, 15.0

devkit_path = 'data/ILSVRC2014/ILSVRC2014_devkit'
det_lists = os.path.join(devkit_path, "data/det_lists")
train_outfile = "train_curated_ratio.txt"

train_det_path = "ILSVRC2014_DET_train"
train_det_bbox_path = "ILSVRC2014_DET_bbox_train"

train_det_img = os.path.join(devkit_path, train_det_path)
train_det_bbox = os.path.join(devkit_path, train_det_bbox_path)

image_ext = '.JPEG'

def get_data_from_tag(node, tag):
    return node.getElementsByTagName(tag)[0].childNodes[0].data
               
def check_image(path):
    xml_path = os.path.join(train_det_bbox, path + '.xml')
    image_path = os.path.join(train_det_img, path + image_ext)
    im = Image.open(image_path)
    
    with open(xml_path) as f:
        data = minidom.parseString(f.read())
        
    sizes = data.getElementsByTagName('size')
    width = int(get_data_from_tag(sizes[0], 'width'))
    height = int(get_data_from_tag(sizes[0], 'height'))
    if im is None:
        print "PIL. Malformed image {}".format(image_path)
        return False
    
    if im.size[0] != width or im.size[1] != height:
        print "Image size wxh {} {}x{} ".format(im.size, width, height)
        return False

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)
    if len(objs) == 0:
        print "No bboxes for image {}".format(image_path)
        return False

    # Load object bounding boxes into a data frame.
    bad_count = 0
    for ix, obj in enumerate(objs):
        x1 = float(get_data_from_tag(obj, 'xmin'))
        y1 = float(get_data_from_tag(obj, 'ymin'))
        x2 = float(get_data_from_tag(obj, 'xmax'))
        y2 = float(get_data_from_tag(obj, 'ymax'))
        if x1 >= x2 or y1 >= y2:
            print "Malformed bounding box wxh:{} {} {} {} {} {}\n{}\n{}".format(
                    width, height, x1, x2, y1, y2, xml_path, image_path)
            return False
        
        w = float(x2 - x1 + 1)
        h = float(y2 - y1 + 1)
        aspect_ratio = w / h
        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
            print "Bad aspect ratio:{} for image\n{}\n{}".format(
                    aspect_ratio, xml_path, image_path)
            bad_count += 1
    
    if bad_count == num_objs:
        print "All objects have bad aspect ratio for image\n{}\n{}".format(
                    xml_path, image_path)
        return False  
    
    return True

def process_train_file(train_list, train_file):
    print "Processing train file {}".format(train_file)
    with open(os.path.join(det_lists,train_file), "r") as f:
        for line in f:
            line = line.strip('\n')
            if check_image(line):
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
#     train_list = train_list[0:100]
    with open(os.path.join(det_lists,train_outfile), "w") as outfile:
        outfile.write("\n".join(train_list))
    

if __name__ == "__main__":
    main()