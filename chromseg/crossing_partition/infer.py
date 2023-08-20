import cv2
from matplotlib import pyplot as plt
import numpy as np
import copy
from skimage import io
import math
import copy
import time
from utils import *

IMG_SIZE = 256
IMAGE_PATH = "/home/truong/datadrive2/project/ChromSeg/dataset/train/data"
OVERLAP_PATH = "/home/truong/datadrive2/project/ChromSeg/region_guided_UNetplus/res_train_0530/output_overlap"
NON_OVERLAP_PATH = "/home/truong/datadrive2/project/ChromSeg/region_guided_UNetplus/res_train_0530/output_non_overlap"
OUTPUT_PATH = './output_res_train_0530'

if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

img_names = os.listdir(IMAGE_PATH)
img_names = [name for name in img_names if 'img' in name.split('_')[-1]]
img_paths = [os.path.join(IMAGE_PATH, name) for name in img_names]
print(img_paths[:5])
IMAGE_SIZE = 256
if __name__ == "__main__":
    for id, img_path in enumerate(img_paths):
        try:
            img_name = img_names[id]
            image = cv2.imread(img_path)
            overlapped = cv2.imread(os.path.join(OVERLAP_PATH, img_name), 0)
            non_overlapped = cv2.imread(os.path.join(NON_OVERLAP_PATH, img_name), 0)
            
            if image.size != (IMAGE_SIZE,IMAGE_SIZE):
                image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            if overlapped.size != (IMAGE_SIZE,IMAGE_SIZE):
                overlapped = overlapped.resize((IMAGE_SIZE, IMAGE_SIZE))
            if non_overlapped.size != (IMAGE_SIZE,IMAGE_SIZE):
                non_overlapped = non_overlapped.resize((IMAGE_SIZE, IMAGE_SIZE))
            
            # overlapped[overlapped == 255] = 1
            # non_overlapped[non_overlapped == 255] = 1
            
            overlapped[overlapped <= 100] = 0
            overlapped[overlapped > 100] = 1
            non_overlapped[non_overlapped <= 100] = 0
            non_overlapped[non_overlapped > 100] = 1

            output = crossing_reconstruct(image, overlapped, non_overlapped)

            output_path = OUTPUT_PATH
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            # cv2.imwrite(os.path.join(output_path, "crossing_" + img_name), image)
            overlapped_vis = cv2.cvtColor(overlapped*255, cv2.COLOR_GRAY2BGR)
            non_overlapped_vis = cv2.cvtColor(non_overlapped*255, cv2.COLOR_GRAY2BGR)
            results = [image, overlapped_vis, non_overlapped_vis]
            try:
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                for idx, mask in enumerate(output):
                    # print(idx)
                    img = copy.deepcopy(gray)
                    new_mask = np.zeros((IMG_SIZE, IMG_SIZE))
                    for i in range(0, IMG_SIZE):
                        for j in range(0, IMG_SIZE):
                            if(mask[i,j] == 1):
                                for x in range(-1,2):
                                    for y in range(-1,2):
                                        try:
                                            new_mask[i+x,j+y] = 1
                                        except:
                                            continue

                    img[(new_mask == 0)] = 255
                    results.append(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR))
                    # cv2.imwrite(os.path.join(output_path, str(num)+".png"), img)
                h_concat = cv2.hconcat(results)
                cv2.imwrite(os.path.join(output_path, "res_" + img_name), h_concat)
            except:
                raise "error(fail to partition)"
        except:
            continue




