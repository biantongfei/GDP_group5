# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Chao Xu (xuchao.19962007@sjtu.edu.cn)
# -----------------------------------------------------

"""API of detector"""
from abc import ABC, abstractmethod


def get_detector(opt=None):
    if opt.detector == 'yolo':
        from Alphapose.detector.yolo_api import YOLODetector
        from Alphapose.detector.yolo_cfg import cfg
        print(YOLODetector(cfg, opt),'---------yolo----------')
        return YOLODetector(cfg, opt)
    elif opt.detector == 'tracker':
        from Alphapose.detector.tracker_api import Tracker
        from Alphapose.detector.tracker_cfg import cfg
        return Tracker(cfg, opt)
    elif opt.detector.startswith('efficientdet_d'):
        from Alphapose.detector.effdet_api import EffDetDetector
        from Alphapose.detector.effdet_cfg import cfg
        return EffDetDetector(cfg, opt)
    else:
        raise NotImplementedError


class BaseDetector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def image_preprocess(self, img_name):
        pass

    @abstractmethod
    def images_detection(self, imgs, orig_dim_list):
        pass

    @abstractmethod
    def detect_one_img(self, img_name):
        pass
