import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import torchvision.models as models
import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
from loss import *
from UNet_plus import UNet_plus2 as UNet_plus2
from utils import *
import argparse
import sys
from utils import x_transforms, y_transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default='1')
    parser.add_argument('--model-type', type=str, default='UNet_plus2')
    parser.add_argument('--dataset', type=str, default='/home/truong/datadrive2/project/ChromSeg/dataset/train/data')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    assert args.model_type in ['UNet', 'UNet_plus', 'UNet_plus2']
    
    model_path = "models/0530_UNet_plus2_original_dataset_seed1_IoU-nan_IoU-0.7401_gamma-1_alpha-0.75.pth"
    IMAGE_SIZE = 256
    batch_size = 8
    
    img_names = os.listdir(args.dataset)
    img_names = [name for name in img_names if 'img' in name.split('_')[-1]]
    img_paths = [os.path.join(args.dataset, name) for name in img_names]
    print(img_paths[:5])

    torch.manual_seed(args.seed)    # reproducible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.model_type == 'UNet_plus2':
        print('UNet_plus2')
        model = UNet_plus2(3, 1)
        # model = model.to(device)
    else:
        raise Exception('Invalid model type: %s'% args.model_type)
    
    # import pdb; pdb.set_trace()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    
    def pre_process(x_path):
        img_x = Image.open(x_path)
        if img_x.size != (IMAGE_SIZE,IMAGE_SIZE):
            img_x = img_x.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_x = x_transforms(img_x)
        img_x = img_x.unsqueeze(0)
        return img_x
    
    

    n = 0
    with torch.no_grad():
        for id, img_path in enumerate(img_paths):
            img_name = img_names[id]
            x = pre_process(img_path).to(device)
            y = model(x)
            y_pred_0 = torch.squeeze(y[0].cpu()).numpy()
            y_pred = np.zeros((256,256))
            y_pred[y_pred_0 > 0.5] = 1.0
            y_2_0 = torch.squeeze(y[1].cpu()).numpy()
            y_2 = np.zeros((256,256))
            y_2[y_2_0 > 0.5] = 1.0
            output1 = np.reshape(y_pred * 255,(256,256))
            output2 = np.reshape(y_2 * 255,(256,256))

            x_image = torch.squeeze(x.cpu()).numpy()
            image = np.dstack((x_image[0,...]*255, x_image[1,...]*255, x_image[2,...]*255))

            cv2.imwrite('res_train_0530/output_overlap/' + img_name, output1)
            cv2.imwrite('res_train_0530/output_non_overlap/' + img_name, output2)
            # cv2.imwrite('mask/' + str(n) + ".png", image)
            