# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.imagenet
from imagenet_eval import voc_eval
import os, sys
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
from PIL import Image

train_det_path = "training/image_2"
train_det_bbox_path = "training/label_2/xml"

val_det_path = "training/image_2"
val_det_bbox_path = "training/label_2/xml"

min_ratio, max_ratio = 0.065, 15.00


class kitti(imdb):
    def __init__(self, image_set, devkit_path):
    
        imdb.__init__(self, image_set)
        self._image_set = image_set
        print "Image set to read images {}".format(self._image_set)
        
        self._train_det_img = os.path.join(devkit_path, train_det_path)
        self._train_det_bbox = os.path.join(devkit_path, train_det_bbox_path)
        self._val_det_img = os.path.join(devkit_path, val_det_path)
        self._val_det_bbox = os.path.join(devkit_path, val_det_bbox_path)
        print self._train_det_img
        print self._train_det_bbox
        print self._val_det_img
        print  self._val_det_bbox
        self._devkit_path = devkit_path
        
#         self._data_path = os.path.join(self._devkit_path, 'data/ILSVRC2True013_DET_' + self._image_set[:-1])
        self._classes = ('__background__', 'car', 'van', 'truck',
                     'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc')
        
        self._class_to_ind = dict(zip(self._classes, xrange(self.num_classes)))
        self._image_ext = ['.png']
        self._anno_ext = '.txt.xml'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # Specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : False,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)

                

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._train_det_img, index + self._image_ext[0])
        if "val" in self._image_set:
            image_path = os.path.join(self._val_det_img, index + self._image_ext[0])
#         image_path = os.path.join(self._data_path,
#                               index[:23] + self._image_ext[0])
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        print "Loading images from {}".format(self._image_set)
        print 
        image_set_file = os.path.join(self._devkit_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_imagenet_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'val2':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print len(roidb)
            with open(cache_file, 'wb') as fid:
                cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
                print 'wrote ss roidb to {}'.format(cache_file)
                return roidb

    def rpn_roidb(self):
        if self._image_set != 'val2':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path, 'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), 'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
        #box_list.append(raw_data[i][:, (1, 0, 3, 2)])
        
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        return the database of selective search regions of interest.
        Ground-truth ROIs are also included.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from txt files of imagenet.
        """
        filename = os.path.join(self._train_det_bbox, index + self._anno_ext)
        image_path = os.path.join(self._train_det_img, index + self._image_ext[0])
        
        if "val" in self._image_set:
            filename = os.path.join(self._val_det_bbox, index + self._anno_ext)
            image_path = os.path.join(self._val_det_img, index + self._image_ext[0])
       
        im = Image.open(image_path)
        if im is None:
            print "PIL. Malformed image {}".format(image_path)
            return
        width = im.size[0]
        height = im.size[1]
            
        # print 'Loading: {}'.format(filename) 
        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        with open(filename) as f:
            data = minidom.parseString(f.read())
            
        sizes = data.getElementsByTagName('size')
       # if im.width != width or im.height != height:
        #    print "Image size wxh {} {}x{} ".format(im.size, width, height)

        objs = data.getElementsByTagName('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        bad_count = 0
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            #x1 = float(get_data_from_tag(obj, 'xmin')) - 1
            #y1 = float(get_data_from_tag(obj, 'ymin')) - 1
            #x2 = float(get_data_from_tag(obj, 'xmax')) - 1
            #y2 = float(get_data_from_tag(obj, 'ymax')) - 1
            x1 = float(get_data_from_tag(obj, 'xmin'))
            y1 = float(get_data_from_tag(obj, 'ymin'))
            x2 = float(get_data_from_tag(obj, 'xmax'))
            y2 = float(get_data_from_tag(obj, 'ymax'))
            class_name = get_data_from_tag(obj, "name").lower().strip()
            if not class_name in self._class_to_ind:
                print "Object {} ignored".format(class_name)
                continue
                
            cls = self._class_to_ind[
                    str(get_data_from_tag(obj, "name")).lower().strip()]
            if x1 >= x2 or y1 >= y2:
                print "Malformed bounding box wxh: {} {} {} {}\n{}\n{}".format(
                        x1, x2, y1, y2, filename, image_path)
                continue
            
            w = float(x2 - x1 + 1)
            h = float(y2 - y1 + 1)
            aspect_ratio = w / h
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                print "Bad aspect ratio:{} for image\n{}\n{}".format(
                    aspect_ratio, filename, image_path)
                bad_count += 1
                continue
            
            if x2 > width - 1: x2 = width - 1 
            if y2 > height - 1 : y2 = height - 1 
            
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
        
        if bad_count == num_objs:
            print "Warning all objs have bad a/r:\n{}\n{}".format(
                    filename, image_path)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}


    def _write_imagenet_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        print "Use salt {}".format(use_salt)
        comp_id = 'comp4'
#         if use_salt:
#             comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
#             print 'Writing {} VOC results file'.format(cls)
            filename = self._get_imagenet_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        
        return comp_id
    
    def _get_imagenet_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = 'comp4' + '_{:s}.txt'
        print filename
        path = os.path.join(
            self._devkit_path, 'results',
            filename)
        return path

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); imagenet_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)
        
    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(self._val_det_bbox, '{:s}' + self._anno_ext)
        print "Anno path {}".format(annopath)
        imagesetfile = os.path.join(
            self._devkit_path, "data/det_lists",
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_imagenet_results_file_template().format(cls)
            print "File name {}".format(filename)
            rec, prec, ap = voc_eval(self._wnid_to_ind, self._class_to_ind,
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        i = 1
        for ap in aps:
            print('{}: {:.3f}'.format(self._classes[i], ap))
            i += 1
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._comp_id = self._write_imagenet_results_file(all_boxes)
        self._do_python_eval(output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.imagenet('val1', '')
    res = d.roidb
    from IPython import embed; embed()
