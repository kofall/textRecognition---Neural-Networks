import skimage
from skimage import img_as_float
from skimage.color import rgb2gray
import imageio
import numpy as np
import random
from os import listdir
from os.path import isfile, join


FNT_PATH = ".\English\Fnt"
DATA_RATIO = 0.9

if __name__ == '__main__':
    output_data = []
    output_labels = []
    test_data = []
    test_labels = []
    counter = 1
    label_num = 1
    directories = [dir for dir in listdir(FNT_PATH) if not(isfile(join(FNT_PATH,dir)))]
    for dir in directories:
        files = [f for f in listdir(join(FNT_PATH,dir)) if isfile(join(join(FNT_PATH,dir),f))]
        
        for file in files:
            print("Loading file nr: "+str(counter)+" at path "+join(join(FNT_PATH,dir),file))
            img_data = rgb2gray(img_as_float(imageio.imread(join(join(FNT_PATH,dir),file))))
            img_array = np.array(255*img_data).astype(np.uint8)
            if random.random() < DATA_RATIO:
                output_data.append(img_array)
                output_labels.append(label_num)
            else:
                test_data.append(img_array)
                test_labels.append(label_num)
            counter+=1

        label_num+=1
    

    output_array = np.array(output_data)
    output_label_array = np.array(output_labels)  
    test_array = np.array(test_data)
    test_label_array = np.array(test_labels)


    np.save('fnt_data.npy',output_array)
    np.save('fnt_data_labels', output_label_array)
    np.save('fnt_test_data.npy', test_array)
    np.save('fnt_test_labels.npy', test_label_array)

