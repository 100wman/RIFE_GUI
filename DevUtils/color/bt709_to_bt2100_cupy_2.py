import glob
import os
import time

import cv2
import numpy as np
import cupy as cp
import result as result

transfer_matrix = cp.array([[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]]).T
# transfer_matrix = cp.expand_dims(cp.expand_dims(transfer_matrix, axis=0), axis=0)  # 1,1, 3,3

def transfer_709_to_2100(i0):
    h, w, c = i0.shape
    i0 = cp.reshape(cp.asarray(i0) / 255., (h*w, c))
    # i0 = cp.expand_dims(i0, axis=3) / 255.  # h,w,c,1
    i0 = cp.power(i0, 2.2)
    # transfer_matrix_repeat = transfer_matrix.repeat(h, axis=0).repeat(w, axis=1)
    result = cp.matmul(i0, transfer_matrix)
    result = pq(result)
    result = cp.asnumpy(cp.reshape(result, (h, w, c)))
    return result

def pq(i1):
    i1 = i1 / 100
    result1 = (cp.power(i1, 0.1593017578125) * 18.8515625) + 0.8359375
    result2 = (cp.power(i1, 0.1593017578125) * 18.6875) + 1
    resultpq = cp.power(result1 / result2, 78.84375) * 255.
    return resultpq

input_dir = r"D:\60-fps-Project\input or ref\Test\ImgSeqTest"
input_list = glob.glob(os.path.join(input_dir, "*.png"))
output_dir = r"D:\60-fps-Project\input or ref\Test\2100_PQ"
os.makedirs(output_dir, exist_ok=True)
for img in input_list:
    start = time.time()
    img_read = cv2.imread(img)
    img_result = transfer_709_to_2100(img_read)
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img)), img_result)
    print(img, time.time() - start)