import copy
import json 
annotation_path = 'C:\\Users\\Daki\\Desktop\\train\\_annotations.coco.json'
image_base_dir = 'C:\\Users\\Daki\\Desktop\\train\\'

annotation_file = open(annotation_path, 'r')
data = annotation_file.read()
data = eval(data)

annotations = {}
for im in data['images']:
    for ann in data['annotations']:
        if im['id'] == ann['image_id']:
            annotations.setdefault(im['id'],[]).append(ann)

# for each image id
for ann in annotations.keys():
    ann_list  = annotations[ann]
    for i in range(int(len(ann_list)/2), len(ann_list)):
        ann_list[i-int(len(ann_list)/2)]['segmentation'] = ann_list[i]['segmentation']
    
    del ann_list[int(len(ann_list)/2):]

data['annotations'] = [ann for k in annotations.keys() for ann in annotations[k]]

with open(annotation_path, 'w') as f:
    json.dump(data, f)

        
    