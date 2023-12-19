from os import listdir, mkdir, path
import shutil
import random

if not path.exists(f'datasets/raabin/train/'):
    mkdir(f'datasets/raabin/train/')
    mkdir(f'datasets/raabin/train/images')
    mkdir(f'datasets/raabin/train/annotations')
    mkdir(f'datasets/raabin/val/')
    mkdir(f'datasets/raabin/val/images')
    mkdir(f'datasets/raabin/val/annotations')

file_annotations = 'datasets\\raabin\\annotations\\annotations.json'
f = open(file_annotations, 'r')
data = f.read()
if data.find('null'):
    data = data.replace('null', 'None')
data = eval(data)
f.close()

files_image = listdir(f'datasets/raabin/images')


labels = {}
for file_image in files_image:
    image_name = file_image.split('.')[0]
    
    obj = data[image_name]

    for cell in range(obj['num_cells']):
        cell_obj = obj[f'cell_{cell}']
        if cell_obj['label2'] not in labels:
            labels[cell_obj['label2']] = [image_name]
        else:
            labels[cell_obj['label2']].append(image_name)
       

keys = list(labels.keys())

print(keys)
for key in keys:
    labels[key] = list(set(labels[key]))
    print(f'{key}:{len(labels[key])}')


nums = {}
for key in keys:
    nums[key] = int(len(labels[key]) * 0.80)


for key in keys:
    for n in range(len(labels[key])):
        if n < nums[key]:
            file_name = data[labels[key][n]]['file_name']
            shutil.copy(f'datasets/raabin/images/{file_name}', f'datasets/raabin/train/images/{file_name}')
        else:
            file_name = data[labels[key][n]]['file_name']
            shutil.copy(f'datasets/raabin/images/{file_name}', f'datasets/raabin/val/images/{file_name}')

shutil.copy(file_annotations, f'datasets/raabin/train/annotations/')
shutil.copy(file_annotations, f'datasets/raabin/val/annotations/')


    


