from os import listdir
import cv2

RIGHT = "RIGHT"
LEFT = "LEFT"

def inside_convex_polygon(point, vertices):
    outside = []
    for p in range(len(vertices)):
        previous_side = None
        n_vertices = len(vertices[p])
        for n in range(n_vertices):
            a, b = vertices[p][n], vertices[p][(n+1)%n_vertices]
            affine_segment = v_sub(b, a)
            affine_point = v_sub(point, a)
            current_side = get_side(affine_segment, affine_point)
            if current_side is None:
                outside.append(True)
                break #outside or over an edge
            elif previous_side is None: #first segment
                previous_side = current_side
            elif previous_side != current_side:
                outside.append(True)
                break
    if len(outside) != len(vertices):
        print('lunghezze diverseeee!!!!!')
    if all(outside) and len(outside) == len(vertices):
        return False
    return True

def get_side(a, b):
    x = cosine_sign(a, b)
    if x < 0:
        return LEFT
    elif x > 0: 
        return RIGHT
    else:
        return None

def v_sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def cosine_sign(a, b):
    return a[0]*b[1]-a[1]*b[0]

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
    cells_data = []
    im_id = data_image['id']
    im = cv2.imread(f'{image_base_dir}\\{data_image["file_name"]}')
    num_cells = len(annotations[im_id])
    l = []
    for cell in range(num_cells):
        cell_data = annotations[im_id][cell]['segmentation'] # [[x,y,x1,y1,...]]
        cells_data.append([(int(cell_data[0][c]),int(cell_data[0][c+1])) for c in range(0,len(cell_data[0]),2)]) # [ [(x,y),(x1,y1),...], [(x,y),(x1,y1),...], ... ] 
    it = im.shape[0] * im.shape[1]
    for i in range(im.shape[1]):
        for j in range(im.shape[0]):
            it-=1
            print(f'{it}/{im.shape[0] * im.shape[1]}', end='\r')
            if inside_convex_polygon((i,j),cells_data):
                im[j][i][1] = 0
            else:
                im[j][i][1] = 255
    im = im[:,:,1]
    cv2.imwrite(f'datasets\\schisto_segm\\train_stuff\\{data_image["file_name"].split(".jpg")[0]}.mask.jpg', im)