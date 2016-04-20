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


# min_allowed_ratio, max_allowed_ratio = 0.07, 15.0
min_allowed_ratio, max_allowed_ratio = 0.065, 15.00
min_ratio, max_ratio = 1000, 0

devkit_path = 'data/kitti-object/training'
labels_path = os.path.join(devkit_path, "label_2/xml")
img_path = os.path.join(devkit_path, "image_2")

train_outfile = "train_kitti.txt"

image_ext = '.png'

def get_data_from_tag(node, tag):
    return node.getElementsByTagName(tag)[0].childNodes[0].data
                
def check_image(path):
    global min_ratio, max_ratio
    xml_path = os.path.join(labels_path, path + '.txt.xml')
    image_path = os.path.join(img_path, path + image_ext)
    im = Image.open(image_path)
     
    with open(xml_path) as f:
        data = minidom.parseString(f.read())
         
#     sizes = data.getElementsByTagName('size')
#     width = int(get_data_from_tag(sizes[0], 'width'))
#     height = int(get_data_from_tag(sizes[0], 'height'))
    if im is None:
        print "PIL. Malformed image {}".format(image_path)
        return False
     
#     if im.size[0] != width or im.size[1] != height:
#         print "Image size wxh {} {}x{} ".format(im.size, width, height)
#         return False
 
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
            print "Malformed bounding box wxh: {} {} {} {}\n{}\n{}".format(
                    x1, x2, y1, y2, xml_path, image_path)
            return False
         
        w = float(x2 - x1 + 1)
        h = float(y2 - y1 + 1)
        aspect_ratio = w / h
        if aspect_ratio < min_allowed_ratio or aspect_ratio > max_allowed_ratio:
            print "Bad aspect ratio:{} for image\n{}\n{}".format(
                    aspect_ratio, xml_path, image_path)
            bad_count += 1
        if aspect_ratio < min_ratio:
            min_ratio = aspect_ratio
        if aspect_ratio > max_ratio:
            max_ratio = aspect_ratio
     
    if bad_count == num_objs:
        print "All objects have bad aspect ratio for image\n{}\n{}".format(
                    xml_path, image_path)
        return False  
     
    return True
 
def process_train_image(train_list, train_file):
#     print "Processing train image {}".format(train_file)
    if check_image(train_file):
        train_list.append(train_file)
 
def filter_train_images(train_images):
    train_list = []
    for train_image in train_images:
        process_train_image(train_list, train_image)
    return train_list

def main():
    train_images = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]
    train_images = [os.path.splitext(f)[0] for f in train_images]
    print train_images
    train_images = filter_train_images(train_images)
    print "Min max ratio:  {} {}".format(min_ratio, max_ratio)

    print len(train_images)
    with open(train_outfile, "w") as outfile:
        outfile.write("\n".join(train_images))
    

if __name__ == "__main__":
    main()