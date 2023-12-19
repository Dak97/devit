from matplotlib import image, transforms
from matplotlib.patches import Rectangle, Polygon
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import cv2
from os import listdir
PLOT_CROP = True
PLOT_ORIG = False
SINGLE_IMAGE = False

file_annotations = 'datasets\\raabin\\annotations\\annotations.json'
f = open(file_annotations, 'r')
data = f.read()
if data.find('null'):
    data = data.replace('null', 'None')
data = eval(data)
f.close()



if SINGLE_IMAGE:
    files_image = ['20160720_232843.jpg']

else:
    files_image = listdir(f'datasets/raabin/images')


for file_image in files_image:
    points = []
    polygons = []
    fig, (ax1, ax2) = plt.subplots(1, 2)

    if PLOT_CROP:
        im = image.imread(f'datasets/raabin/images/{file_image}')

        obj = data[f'{file_image.split(".")[0]}']
        
        num_cells = obj['num_cells']
        print(f'image {file_image}, num_cell {num_cells}')

        cells_obj = []
        for cell in range(num_cells):
            print(f'{obj[f"cell_{cell}"]["label2"]}')
            cells_obj.append(obj[f'cell_{cell}'])
            
        
        bb_points = [(cell['x1'], cell['y1'], cell['x2'], cell['y2']) for cell in cells_obj]

        mask_points_x_list = [cell['x_points'] for cell in cells_obj]
        mask_points_y_list = [cell['y_points'] for cell in cells_obj]

        for x_points, y_points in zip(mask_points_x_list, mask_points_y_list):
                polygons.append([(x,y) for x,y in zip(x_points,y_points)])

        # mask_points = [(x,y) for x_ps,y_ps in zip(mask_points_x_list, mask_points_y_list) for x,y in zip(x_ps, y_ps)]
        # print(mask_points)
        
        
        

    if PLOT_ORIG:
        im_or = image.imread(f'dataset/images/{file_image}')
        # # im_crop = Image.open(f'images/{file_image}')

        f_json = open(f'dataset/jsons/{file_json}.json', 'r')
        data = f_json.read()
        if data.find('null'):
            data = data.replace('null', 'None')
        data_json = eval(data)

        f_json.close()


        num_cell = data_json['Cell Numbers']
        print(f'Numero di cellule: {num_cell}')
        cell_coord = []
        for i in range(num_cell):
            coord = (int(data_json[f'Cell_{i}']['x1']), int(data_json[f'Cell_{i}']['y1']), int(data_json[f'Cell_{i}']['x2']), int(data_json[f'Cell_{i}']['y2']))
            cell_coord.append(coord)




        for cell in cell_coord:
            p1 = [cell[0], cell[1]]
            p2 = [cell[2], cell[3]]

            p3 = [p2[0], p1[1]]
            p4 = [p1[0], p2[1]]

            x1 = [p1[0], p3[0]]
            y1 = [p1[1], p3[1]]
            x2 = [p3[0], p2[0]]
            y2 = [p3[1], p2[1]]
            x3 = [p2[0], p4[0]]
            y3 = [p2[1], p4[1]]
            x4 = [p4[0], p1[0]]
            y4 = [p4[1], p1[1]]

            print(p1, p2, im_or.shape)
            


            xc = int ((p1[0] + p3[0]) / 2)
            yc = int ((p1[1] + p4[1]) / 2)

            ax1.plot(xc, yc, marker='.', color='white')
            # ax1.plot(x1, y1,x2, y2,x3, y3, x4, y4, marker='.', color='red')

        ax1.imshow(im_or)
    # plt.show()
    if PLOT_CROP:
        for cell in bb_points:
            ax2.plot(cell[0], cell[1], cell[2], cell[3], marker='.', color='blue')
            ax2.add_patch(Rectangle((cell[0],cell[1]), cell[2]-cell[0], cell[3]-cell[1], facecolor='none', edgecolor='red'))
        for p in range(len(polygons)):
            polygon = Polygon(polygons[p])
            ax2.add_patch(polygon)
        ax2.imshow(im)
        
    plt.show()
    
    
    