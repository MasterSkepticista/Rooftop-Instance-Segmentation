import os
import PIL
from PIL import Image
import cv2
import numpy as np

target_size = (1792, 1792)

list = os.listdir('image')
os.makedirs('resized')
print(len(list))
i = 1
for item in list:

    print('Converting...{0}'.format(i))
    name = os.path.basename(item)
    image = np.array(Image.open(os.path.join('image',name)))
    name = os.path.splitext(name)
    resized = cv2.resize(image, target_size)
    cv2.imwrite(os.path.join('resized',name[0]+'.png'), resized)
    i += 1

print('Finished convert')

