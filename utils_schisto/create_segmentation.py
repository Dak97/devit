import cv2
import numpy as np

annotation_path = 'datasets\\schisto\\train_prova\\annotations.json'
image_base_dir = 'datasets\\schisto\\train_prova\\'

annotation_file = open(annotation_path, 'r')
data = annotation_file.read()
data = eval(data)

images = data['images']

annotations = {}
for im in data['images']:
    for ann in data['annotations']:
        if im['id'] == ann['image_id']:
            annotations.setdefault(im['id'],[]).append(ann)

for image in images:
    name = image['file_name']
    im = cv2.imread(image_base_dir + name, cv2.IMREAD_GRAYSCALE)

    
    # im = im[:,:,1]
    im = im * (im<175) 
    bboxes = annotations[image['id']]
    bbox = bboxes[0]['bbox']

    
    for i in range(im.shape[1]):
        for j in range(im.shape[0]):
            if not (i >= bbox[0] and i <= bbox[2] + bbox[0] and j >= bbox[1] and j <= bbox[3] +  bbox[1]):
                im[j][i] = 0

    cv2.namedWindow('image')
    cv2.imshow('image', im)
    cv2.waitKey(0)

    result = np.zeros_like(im)
    contours = cv2.findContours(im , cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours[0]]
    
    segm = [c for i in contours[0][areas.index(max(areas))].tolist() for j in i for c in j]
    cv2.drawContours(result, [contours[0][areas.index(max(areas))]], 0, 255, 1)
    # for cntr in contours[0]:
    #     area = cv2.contourArea(cntr)
    #     if area > 20:
    #         segm = [c for i in cntr.tolist() for j in i for c in j]
    #         cv2.drawContours(result, [cntr], 0, 255, 1)

    cv2.namedWindow('image')
    cv2.imshow('image', result)
    cv2.waitKey(0)