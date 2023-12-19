from os import listdir
import cv2

def inside_bb(x1,y1,x2,y2,i,j):
    for _x1, _x2,_y1,_y2 in zip(x1,x2,y1,y2):
       if i >= _x1 and i <= _x2 and j >= _y1 and j <= _y2:
           return True
    return False

annotation_path = 'datasets\\schisto_segm\\train_stuff\\annotations.json'
image_base_dir = 'datasets\\schisto_segm\\train_stuff\\'

annotation_file = open(annotation_path, 'r')
data = annotation_file.read()
data = eval(data)

# files_image = listdir(image_base_dir)
# files_image = [im for im in files_image if '.jpg' in im]

# files_image = files_image[:]

images = data['images']

annotations = {}
for im in data['images']:
    for ann in data['annotations']:
        if im['id'] == ann['image_id']:
            annotations.setdefault(im['id'],[]).append(ann)
            
for data_image in images[:]:
    x1, y1, x2, y2 = [], [], [], []
    im_id = data_image['id']
    im = cv2.imread(f'{image_base_dir}\\{data_image["file_name"]}')
    num_cells = len(annotations[im_id])
    
    for cell in range(num_cells):
        cell_data = annotations[im_id][cell]['bbox']
        x1.append(cell_data[0])
        y1.append(cell_data[1])
        x2.append(cell_data[0] + cell_data[2])
        y2.append(cell_data[1] + cell_data[3])

    for i in range(im.shape[1]):
        for j in range(im.shape[0]):
    
            if inside_bb(x1,y1,x2,y2,i,j):
                im[j][i][1] = 0
            else:
                im[j][i][1] = 1
    im = im[:,:,1]
    cv2.imwrite(f'datasets\\schisto_segm\\train_stuff\\{data_image["file_name"].split(".jpg")[0]}.mask.jpg', im)