import cv2
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
import random

def register_schisto_dataset(dataset_name,annotations_file_path, images_dir):
    register_coco_instances(dataset_name, {}, annotations_file_path, images_dir)

if __name__ == '__main__':

    split = 'test'
    
    register_schisto_dataset(f'schisto_{split}',f'datasets\\schisto_segm\\{split}\\annotations.json', f'datasets\\schisto_segm\\{split}\\')
    # register_schisto_dataset(f'schisto_{split}',f'C:\\Users\\Daki\\Desktop\\train\\_annotations.coco.json', f'C:\\Users\\Daki\\Desktop\\train\\')

    schisto_metadata = MetadataCatalog.get(f'schisto_{split}')
    
    dataset_dicts = DatasetCatalog.get(f"schisto_{split}")
    
    # for d in random.sample(dataset_dicts, 10):
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        name = d["file_name"].split('\\')[-1]
        visualizer = Visualizer(img[:, :, ::-1], metadata=schisto_metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)
        cv2.imwrite(f'datasets\\schisto_segm\\test_gt\\{name}', out.get_image()[:, :, ::-1])
        # cv2.imshow('Schisto',out.get_image()[:, :, ::-1])
        # cv2.waitKey(0)
