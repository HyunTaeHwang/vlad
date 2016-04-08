#!/usr/bin/env python

import _init_paths
import os
import xml.dom.minidom as minidom
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


min_ratio, max_ratio = 100000, 0

devkit_path = 'data/VOCdevkit2007'
annotations_path = os.path.join(devkit_path, "VOC2007/Annotations")

def get_data_from_tag(node, tag):
    return node.getElementsByTagName(tag)[0].childNodes[0].data
               
def process_train_file(path):
    global min_ratio, max_ratio
#     print "Processing train file {}".format(path)
    xml_path = os.path.join(annotations_path, path)
    
    with open(xml_path) as f:
        data = minidom.parseString(f.read())
        
    sizes = data.getElementsByTagName('size')
    width = int(get_data_from_tag(sizes[0], 'width'))
    height = int(get_data_from_tag(sizes[0], 'height'))
   
    objs = data.getElementsByTagName('object')
    if len(objs) == 0:
        print "No bboxes for image {}".format(path)
        return

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        x1 = float(get_data_from_tag(obj, 'xmin'))
        y1 = float(get_data_from_tag(obj, 'ymin'))
        x2 = float(get_data_from_tag(obj, 'xmax'))
        y2 = float(get_data_from_tag(obj, 'ymax'))
        if x1 >= x2 or y1 >= y2:
            print "Malformed bounding box wxh:{} {} {} {} {} {}\n{}".format(
                    width, height, x1, x2, y1, y2, xml_path)
            return
      
        w = float(x2 - x1 + 1)
        h = float(y2 - y1 + 1)
        aspect_ratio = w / h
        if aspect_ratio < min_ratio:
            min_ratio = aspect_ratio
        if aspect_ratio > max_ratio:
            max_ratio = aspect_ratio

def process_train_files(train_files):
    for train_file in train_files:
        process_train_file(train_file)


def main():
    train_txts = [f for f in os.listdir(annotations_path) if os.path.isfile(os.path.join(annotations_path, f))]
    train_files = []
    for train in train_txts:
        train_files.append(train)
    
    print train_files
    process_train_files(train_files)
    print "Min, max aspect ratio {} {}".format(min_ratio, max_ratio)
    

if __name__ == "__main__":
    main()