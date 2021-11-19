import glob
import os
import time

import cv2
import numpy as np
from line_profiler_pycharm import profile
import torch
transfer_matrix = np.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]])
device = torch.device('cuda')
transfer_matrix = torch.from_numpy(transfer_matrix).T.to(device, non_blocking=True).float()


@profile
def transfer_709_to_2020(i0):
    h, w, c = i0.shape
    dt = i0.dtype
    i0 = torch.from_numpy(i0).to(device,non_blocking=True).float() / 255.  # h,w,3
    i0 = torch.reshape(i0, (h*w, c))
    result = torch.matmul(i0, transfer_matrix)  # h*w,3 * 3,3
    result = (result * 255.).byte().cpu().numpy().copy()
    result = np.reshape(result, (h, w, c)).astype(dt)
    return result

input_dir = r"D:\60-fps-Project\input or ref\Test\ImgSeqTest"
input_list = glob.glob(os.path.join(input_dir, "*.png"))
output_dir = r"D:\60-fps-Project\input or ref\Test\output\bt2020"
os.makedirs(output_dir, exist_ok=True)
for img in input_list:
    start = time.time()
    img_read = cv2.imread(img)
    img_result = transfer_709_to_2020(img_read)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img)), img_result)
    print(img, time.time() - start)