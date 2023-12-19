#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.multiprocessing
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')


import lib.data.fewshot
import lib.data.ovdshot
from lib.categories import SEEN_CLS_DICT, ALL_CLS_DICT

from collections import defaultdict

import numpy as np

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS, \
    COCO_UNSEEN_CLS, COCO_OVD_ALL_CLS
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2 import model_zoo
    
from sklearn.metrics import precision_recall_curve
from sklearn import metrics as sk_metrics

from utils.load_dataset import get_WBC_dicts
import random
import cv2
from detectron2.utils.visualizer import Visualizer

from detectron2.engine import HookBase
from matplotlib import pyplot as plt
from matplotlib import image as im_
import warnings
warnings.simplefilter(action='ignore',category=UserWarning)
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []

        if 'OpenSet' in cfg.MODEL.META_ARCHITECTURE:
            if 'lvis' in dataset_name:
                evaluator_list.append(LVISEvaluator(dataset_name, output_dir=output_folder))
            else:
                dtrain_name = cfg.DATASETS.TRAIN[0]
                seen_cnames = SEEN_CLS_DICT[dtrain_name]
                all_cnames = ALL_CLS_DICT[dtrain_name]
                unseen_cnames = [c for c in all_cnames if c not in seen_cnames]
                evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder, few_shot_mode=True,
                                                    seen_cnames=seen_cnames, unseen_cnames=unseen_cnames, all_cnames=all_cnames))
            return DatasetEvaluators(evaluator_list)
    

        # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        evaluator_type = None
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )

        if dataset_name == 'schisto_val':
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if dataset_name == 'schisto_test':
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if dataset_name == 'WBC_val':
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args, is_test):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # import ipdb; ipdb.set_trace()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.DATASETS.TRAIN = ("WBC_train",)
    # cfg.DATASETS.TEST = ("WBC_val",)

    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.STEPS = (333,444)
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.CHECKPOINT_PERIOD = 20

    if is_test:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    
    cfg.freeze()
    default_setup(cfg, args)
    
    return cfg

def calc_IoU(pred, gt):
    x_left = max(pred[0], gt[0])
    y_top = max(pred[1], gt[1])
    x_right = min(pred[2], gt[2])
    y_bottom = min(pred[3], gt[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    bb2_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou
class IterHook(HookBase):
    def __init__(self) -> None:
        self.start_time = time.time() 
        
    def before_step(self):
        elapsed_time = time.time() - self.start_time
        # estimate = (elapsed_time / (self.trainer.iter)+1) * self.trainer.max_iter 
        
        print(f'Start: {self.trainer.start_iter}, Iter:{self.trainer.iter}, End: {self.trainer.max_iter} \
                Remain time:{0}', end='\r')
    # def after_step(self):
    #     for k, v in self.trainer.storage.histories().items():
    #         if ("loss" in k) or ("acc" in k) or ("ratio" in k) or ("rec" in k):
    #             print(f'{k}, {v.median(1)}')


def main(args):

    TRAIN = False
    TEST = not TRAIN
    VIS = False
    
    cfg = setup(args, TEST)
    # print(cfg)

    
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    
    if VIS:
        train_loader = Trainer.build_train_loader(cfg)
        for sample_image_batch_idx, train_image_batch in enumerate(train_loader):
            for idx, train_image in enumerate(train_image_batch):
                fig, (ax1, ax2) = plt.subplots(1, 2)
                im_or = im_.imread(train_image['file_name'])
                ax1.imshow(im_or)
                
                image = train_image["image"].numpy().transpose(1, 2, 0)
                # visualize ground truth
                gt_visualizer = Visualizer(
                    image[:, :, ::-1], metadata=MetadataCatalog.get('WBC_train'), scale=1
                )
                gt_image = gt_visualizer.overlay_instances(
                            boxes=train_image["instances"].gt_boxes, 
                            labels=train_image["instances"].gt_classes
                        )
                
                ax2.imshow(gt_image.get_image()[:, :, ::-1])
                
                # cv2.imshow('images', gt_image.get_image()[:, :, ::-1])
                # cv2.waitKey(0)
                plt.savefig(f'C:\\Users\\Daki\\Desktop\\prova5\\devit\\datasets\\raabin\\augmentations\\{train_image["file_name"].split("/")[-1]}')
                plt.close(fig)
    
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    if TRAIN:
        trainer = Trainer(cfg)
        trainer.register_hooks(hooks=[IterHook()])
        
        trainer.resume_or_load(resume=args.resume)
        
        if cfg.TEST.AUG.ENABLED:
            trainer.register_hooks(
                [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
            )
        
        return trainer.train()

    if TEST:    
        predictor = DefaultPredictor(cfg)

        # test = get_WBC_dicts('datasets/raabin/val')
        false_neg = 0
        count_cell = 0
        FN = False
        dataset_dicts = DatasetCatalog.get(f"{cfg.DATASETS.TEST[0]}")
        for d in dataset_dicts:
            
            count_cell += len(d['annotations'])
            im = cv2.imread(d['file_name'])
            
            print(d['file_name'])
            outputs = predictor(im)
            # _, i = torch.max(outputs['instances'].get('scores'),0)
            # if outputs['instances'].get('scores').shape[0] >= 3:
            #     outputs['instances'] = outputs['instances'][0:2]
            
            print(outputs["instances"])
            if FN:
                for bb in [bbox['bbox'] for bbox in d['annotations']]: # bb = [x,y,w,h]
                    FN_flag = True
                    bb = [bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]] # bb = [x,y,x1,y1]
                    for bb_pred in outputs['instances'].get('pred_boxes').tensor.tolist():
                        # calc IoU
                        iou = calc_IoU(bb_pred, bb)
                        if iou > 0.5: # una predizione copre il almeno il 50 % del gt
                            FN_flag = False # non c'e un falso negativo per quel bb
                            break
                    
                    if FN_flag:
                        false_neg += 1 
                print(f'FN: {false_neg}/{count_cell}')
            
            v = Visualizer(im[:, :, ::-1],
                  metadata=MetadataCatalog.get(f'{cfg.DATASETS.TEST[0]}'), 
                   scale=1
                   )   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # out = v.draw_dataset_dict(d)
        
            if not os.path.isdir(f'{str(cfg.OUTPUT_DIR).split(".")[1][1:]}\\th_{int(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST*100)}'):
                os.mkdir(f'{cfg.OUTPUT_DIR}\\th_{int(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST*100)}')

            
            if 'WBC' in cfg.DATASETS.TEST[0]:
                name = d['file_name'].split('/')[-1] # WBC
            else:
                name = d['file_name'].split('\\')[-1] # schisto
            cv2.imwrite(f'{cfg.OUTPUT_DIR}\\th_{int(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST*100)}\\{name}', out.get_image()[:, :, ::-1])
           
    

        


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )