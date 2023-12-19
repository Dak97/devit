from os import listdir
import cv2

def inside_bb(x1,y1,x2,y2,i,j):
    for _x1, _x2,_y1,_y2 in zip(x1,x2,y1,y2):
       if i >= _x1 and i <= _x2 and j >= _y1 and j <= _y2:
           return True
    return False

annotation_path = 'datasets\\raabin\\val\\annotations\\annotations.json'
image_base_dir = 'datasets\\raabin\\val\\images'

annotation_file = open(annotation_path, 'r')
data = annotation_file.read()
annotations = eval(data)

files_image = listdir(image_base_dir)

files_image = files_image[:]

for file in files_image:
    x1, y1, x2, y2 = [], [], [], []
    im_annotation = annotations[file.split('.')[0]]
    num_cells = im_annotation['num_cells']
    im = cv2.imread(f'{image_base_dir}\\{file}')
    for cell in range(num_cells):
        cell_data = im_annotation[f'cell_{cell}']
        x1.append(cell_data['x1'])
        y1.append(cell_data['y1'])
        x2.append(cell_data['x2'])
        y2.append(cell_data['y2'])

    for i in range(im.shape[1]):
        for j in range(im.shape[0]):
    
            if inside_bb(x1,y1,x2,y2,i,j):
                im[j][i][1] = 0
            else:
                im[j][i][1] = 1
    im = im[:,:,1]
    cv2.imwrite(f'datasets\\raabin\\val\\masks\\{file.split(".")[0]}.mask.jpg', im)