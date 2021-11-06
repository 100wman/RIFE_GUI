import glob
import os
import time

import cv2
import numpy as np
from line_profiler_pycharm import profile
import torch
transfer_matrix = np.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]])
device = torch.device('cuda')
w,h = 1920, 1080
transfer_matrix = torch.from_numpy(transfer_matrix).to(device, non_blocking=True).float().repeat(h,w,1,1)  # h,w,3,3


@profile
def transfer_709_to_2020(i0):
    i0 = torch.from_numpy(i0).to(device,non_blocking=True).float().unsqueeze(3) / 255.  # h,w,3,1
    result = transfer_matrix @ i0  # h,w,3,3 @ h,w,3,1
    result = (result * 255.).squeeze(3).byte().cpu().numpy().copy()
    return result

input_dir = r"D:\60-fps-Project\input or ref\Test\output\00907_1bceb2_165961"
input_list = glob.glob(os.path.join(input_dir, "*.png"))
output_dir = r"D:\60-fps-Project\input or ref\Test\output\bt2020"
os.makedirs(output_dir, exist_ok=True)
for img in input_list:
    start = time.time()
    img_read = cv2.imread(img)
    img_result = transfer_709_to_2020(img_read)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img)), img_result)
    print(img, time.time() - start)