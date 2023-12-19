from os import listdir, system
from PIL import Image, ImageDraw, ImageOps
import cv2
import numpy as np
import json
import copy

BASE_DIR = 'C:\\Users\\Daki\\Desktop\\UNIVERSITA\\LODDO\\prova_gpu\\CrowdCounting-P2PNet\\'
# files = listdir(f'{BASE_DIR}dataset\\images')
files = listdir(f'datasets/raabin/val/images')
# files = files[0:500]

ind = 0
m = len(files)
print(m)
tot_cells = {}


histogram_classes = {}
for file in files:
    ind = ind + 1

    print(f'{ind}/{m} - {file}', end='\r')
    
    # im = cv2.imread(f'{BASE_DIR}dataset\\images\\{file}')
    

    # f = open(f'{BASE_DIR}dataset\\jsons\\{file.split(".")[0]}.json', 'r')
    f = open(f'datasets/raabin/val/annotations/annotations.json', 'r')

    data = f.read()
    if data.find('null'):
        data = data.replace('null', '\'Unknn\'')
    data = eval(data)
    f.close()

    # num_cell = data['Cell Numbers']
    num_cell = data[f'{file.split(".")[0]}']['num_cells']
    # print(num_cell)
    cell_coord = []
    cell_label = []
    error_label = False
    for i in range(num_cell):
        
        # label = data[f'Cell_{i}']['Label1']
        label = data[f'{file.split(".")[0]}'][f'cell_{i}']['label1']
       

        if label not in histogram_classes:
            histogram_classes[label] = 1
        else:
            histogram_classes[label] += 1

        cell_label.append(label)
    f.close
print(histogram_classes)