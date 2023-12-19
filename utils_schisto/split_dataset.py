import os
import shutil
import json
import copy
import cv2
import numpy

def resize(file_name, annotations, scale):
    im = cv2.imread(f'{BASE_DIR}{file_name}')

    # image resize
    width = int(im.shape[1] * scale / 100)
    height = int(im.shape[0] * scale / 100)
    dim = (width, height)
    im_res = cv2.resize(im, dim)


    im_id = 0
    anns = []
    for item in annotations['images']:
        if item['file_name'] == file_name:
            item['height'] = height
            item['width'] = width
            im_id = item['id']
            break
    
    for item in annotations['annotations']:
        if item['image_id'] == im_id:
            item['bbox'] = [
                int(item['bbox'][0] * scale / 100),
                int(item['bbox'][1] * scale / 100),
                int(item['bbox'][2] * scale / 100),
                int(item['bbox'][3] * scale / 100),
            ]
            item['area'] = round(item['area'] * scale / 100, 3)
            x1, y1= item['bbox'][0], item['bbox'][1]
            x2, y2 = x1 + item['bbox'][2], y1 + item['bbox'][3]

            item['segmentation'] = [[round(c * scale / 100, 2) for c in item['segmentation'][0]]]
            # extraction segmentation points

            # x_ce = int ((x1 + x2) / 2)
            # y_ce = int ((y1 + y2) / 2)

            # pad = 0
            
            
            # ys = []
            
            # r = x_ce - x1 if x_ce - x1 < y_ce - y1 else  y_ce - y1

            # xs = numpy.arange(x_ce - r, x_ce + r).tolist()

            # for x in xs:
            #     y = int(numpy.sqrt(r**2 - (x - x_ce)**2) + y_ce)
            #     ys.append(y)
            # for x in xs:
            #     y = int(-numpy.sqrt(r**2 - (x - x_ce)**2) + y_ce)
            #     ys.append(y)
            
            # xs = xs + xs[::-1]

            # poly = [xy_ for xy in zip(xs,ys) for xy_ in xy]

            # item['segmentation'] = [poly]

    return im_res, annotations
    
BASE_DIR = 'datasets\\Schisto_COCO_segm\\train\\'
DATASET_DIR = 'datasets\\schisto_segm\\'

ANNOTATIONS_PATH = 'datasets\\Schisto_COCO_segm\\train\\_annotations.coco.json'

annotations = {}
with open(ANNOTATIONS_PATH, 'r') as annotations_file:
    data = annotations_file.read()
    data = eval(data)
    annotations = data

images_data = annotations['images']
images_annotations = annotations['annotations']
# annotations['categories'][0]['id'] = 1
# annotations['categories'][1]['id'] = 0

histogram = {}
for image_data in images_data:
    image_name = image_data['file_name']
    image_id = image_data['id']

    histogram[image_name] = 0
    for annotation in images_annotations:
        if annotation['image_id'] == image_id:
            annotation['category_id'] = 1
            histogram[image_name] += 1


# create train set
train, val, test = [], [], []
for im in histogram.keys():
    if histogram[im] > 7:
        train.append(im)
        # print(f'{im} - {histogram[im]}')

# create validation set
for i in range(1,8):
    for im in histogram.keys():
        if histogram[im] == i:
            val.append(im)
            break
# create test set         
for im in histogram.keys():
    if im not in train and im not in val:
        if histogram[im] != 0:
            test.append(im)

# CREATE DATASET
train_path = f'{DATASET_DIR}train\\'
val_path = f'{DATASET_DIR}val\\'
test_path = f'{DATASET_DIR}test\\'
# if not exist create dirs
if not os.path.isdir(f'{DATASET_DIR}train'):
    os.mkdir(train_path)
    os.mkdir(val_path)
    os.mkdir(test_path)

# CREATE ANNOTATIONS FILE FOR TRAIN, VAL AND TEST
annotation_train = copy.deepcopy(annotations)
annotation_val = copy.deepcopy(annotations)
annotation_test = copy.deepcopy(annotations)

for item in annotations['images']:
    if item['file_name'] not in train:
        annotation_train['images'].remove(item)
for item in annotations['images']:
    if item['file_name'] not in val:
        annotation_val['images'].remove(item)
for item in annotations['images']:
    if item['file_name'] not in test:
        annotation_test['images'].remove(item)

# RESIZE IMAGE and ANNOTATIONS
scale = 30
for im in train:
    im_res, annotation_train = resize(im,annotation_train,scale)
    cv2.imwrite(f'{train_path}{im}', im_res)
for im in val:
    im_res, annotation_val = resize(im,annotation_val,scale)
    cv2.imwrite(f'{val_path}{im}', im_res)
for im in test:
    im_res, annotation_test = resize(im,annotation_test,scale)
    cv2.imwrite(f'{test_path}{im}', im_res)


with open(f'{train_path}annotations.json', 'w') as f:
    json.dump(annotation_train, f)
with open(f'{val_path}annotations.json', 'w') as f:
    json.dump(annotation_val, f)
with open(f'{test_path}annotations.json', 'w') as f:
    json.dump(annotation_test, f)
   

# copy images in relative folder
# for im in train:
#     shutil.copy(f'{BASE_DIR}{im}', train_path)
# for im in val:
#     shutil.copy(f'{BASE_DIR}{im}', val_path)
# for im in test:
#     shutil.copy(f'{BASE_DIR}{im}', test_path)


    
