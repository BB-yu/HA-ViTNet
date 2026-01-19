import json
import cv2
import numpy as np

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

import os
import glob
files=glob.glob(' ')



for nam in files:

    1

    name=os.path.basename(nam).split('.')[0]

    json_file_path1 = nam.split('\\')[:-2] +[' ']+[nam.split('\\')[-1].replace('JPG','json')]

    
    json_file_path='\\'.join(json_file_path1)

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    image_path = data['imagePath']

    image_path=nam



    label_colors = {
        'road': (0, 1, 0, 0.3),
        'plant': (1, 0, 0, 0.3)
        
    }



    image = Image.open(image_path)


    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)


    for shape in data['shapes']:
        points = shape['points']
        label = shape['label']

        points = np.array(points)

        color = label_colors.get(label, (0, 0, 1, 0.5))

        polygon = patches.Polygon(points, closed=True, edgecolor=color, facecolor=color, linewidth=2)
        ax.add_patch(polygon)

        centroid = np.mean(points, axis=0)
        color1=list(color[:3])+[1]

        ax.text(centroid[0], centroid[1], label, fontsize=12, color=color1, weight='bold')


    plt.axis('off')
    plt.savefig(r" "+os.path.basename(nam))
    plt.show()

