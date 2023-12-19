from os import listdir, path
import cv2
from json import load
import numpy as np
import torch
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import random

def get_WBC_dicts(im_dir):
    # TODO creare un dict che converta la label in int
    # TODO 'Neutrophil', 'Unknown', 'Monocyte', 'Burst', 'Large Lymph', 'Small Lymph', 'Eosinophil', 'Artifact'

    label_to_id = {
        'WBC': 0, 'Others': 1
    }
    images = listdir(f'{im_dir}/images')
    annotations_file = f'{im_dir}/annotations/annotations.json'

    with open(annotations_file) as f:
            data = eval(f.read())

    dataset_dicts = []
    idx = 0
    for image in images:
        record = {}

        
        path_image = f'{im_dir}/images/{image}'
        image_data = data[image.split('.')[0]]

        height, width = cv2.imread(path_image).shape[:2]

        record['file_name'] = path_image
        record['image_id'] = idx
        record['height'] = height
        record['width'] = width

        num_cells = image_data['num_cells']

        objs = [] # contiene tutte le cellule annotate presenti in una immagine
        for cell in range(num_cells):
            cell_data = image_data[f'cell_{cell}']

            pxs = cell_data['x_points']
            pys = cell_data['y_points']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(pxs, pys)]
            poly = [p for x in poly for p in x]

            obj = {
                'bbox': [cell_data['x1'],cell_data['y1'],cell_data['x2'],cell_data['y2']],
                'bbox_mode': BoxMode.XYXY_ABS,
                "segmentation": [poly],
                'category_id':  0 #label_to_id[cell_data['label2']]
            }
            
            objs.append(obj)
        
        record['annotations'] = objs
        record['sem_seg'] = torch.tensor(cv2.imread(f'{im_dir}/masks/{image.split(".")[0]}.mask.jpg'))[:,:,1]
        dataset_dicts.append(record)

    return dataset_dicts
def register_dataset(dataset_name, dir='datasets/raabin/', split=['train', 'val'], labels=['Neutrophil', 'Burst',  'Monocyte', 'Unknown', 'Large Lymph', 'Small Lymph', 'Eosinophil', 'Artifact']):
    for d in split:
        DatasetCatalog.register(dataset_name + d, lambda d=d: get_WBC_dicts(dir+d))
        MetadataCatalog.get(dataset_name+d).set(thing_classes=labels)
        # MetadataCatalog.get(dataset_name+d).set(stuff_classes=['bb', 'bg'])


def get_balloon_dicts(img_dir):
   
    i = 0
    dataset_dicts = []
    for cat in listdir(img_dir):
        
        # json_file = path.join(img_dir, "via_region_data.json")
        json_file = f'{img_dir+cat}/via_region_data.json'
        with open(json_file) as f:
            imgs_anns = load(f)

        
        for idx, v in enumerate(imgs_anns.values()):
            record = {}
            if v['filename'] in listdir(img_dir+cat):
                filename = path.join(img_dir+cat, v["filename"])
                height, width = cv2.imread(filename).shape[:2]
                
                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = height
                record["width"] = width
            
                annos = v["regions"]
                objs = []
                for _, anno in annos.items():
                    assert not anno["region_attributes"]
                    anno = anno["shape_attributes"]
                    px = anno["all_points_x"]
                    py = anno["all_points_y"]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]

                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": i,
                    }
                    objs.append(obj)
                record["annotations"] = objs
                dataset_dicts.append(record)
        i += 1
    return dataset_dicts


if __name__ == '__main__':

    register_dataset('WBC_', split=['train', 'val'], labels=['WBC', 'Other'])

    WBC_metadata = MetadataCatalog.get('WBC_val')
    
    dataset_dicts = get_WBC_dicts('datasets/raabin/val')
    
    # for d in random.sample(dataset_dicts, 10):
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=WBC_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow('WBC',out.get_image()[:, :, ::-1])
        cv2.waitKey(0)




