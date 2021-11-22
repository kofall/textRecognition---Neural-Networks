import skimage
from skimage import img_as_float
from skimage.color import rgb2gray
import imageio
import numpy as np
from os import listdir
from os.path import isfile, join

from skimage.util.dtype import img_as_uint

FNT_PATH = "./English/Fnt/"

if __name__ == '__main__':
    output_data = []
    output_labels = []
    directories = [dir for dir in listdir(FNT_PATH) if not(isfile(join(FNT_PATH,dir)))]
    for dir in directories:
        files = [f for f in listdir(join(FNT_PATH,dir)) if isfile(join(join(FNT_PATH,dir),f))]
        for file in files:
            output_data.append(rgb2gray(img_as_uint(join(join(FNT_PATH,dir),file))))
    
    output_array = np.array(output_data)
    np.save('fnt_data.npy',output_array)

