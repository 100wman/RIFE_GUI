import argparse
from utils.esrlupscale import RRDBNetUpscaler
import cv2
import math
from queue import Queue
from tqdm import tqdm
import numpy as np
import os
import torch
import warnings
from utils import esrlupscale
import _thread
import sys
from basicsr.archs.rrdbnet_arch import RRDBNet
import time
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./models/RealESRGAN_x4plus.pth',help='超分模型')
parser.add_argument('--mod_scale', type=int, default=4,help='缩放倍数,先不要更改')
parser.add_argument('--input', type=str, default='input', help='输入文件夹')
parser.add_argument('--output', type=str, default='output', help='输出文件夹')
parser.add_argument('--precent', type=float, default=90,help = '显存使用百分比')
parser.add_argument('--gpu_id', type=int, default=0, help='显卡ID (N卡) ')
args = parser.parse_args()

device = torch.device("cuda")
torch.cuda.set_device(args.gpu_id)

def build_read_buffer(user_args, read_buffer, files):
    try:
        for frame in files:
            frame = cv2.imdecode(np.fromfile(os.path.join(user_args.input, frame), dtype=np.uint8), 1)[:, :, ::-1].copy()
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

cnt = 1
def clear_write_buffer(write_buffer,files):
    global cnt
    while True:
        item = write_buffer.get()
        if item is None:
            break
        cv2.imencode('.png', item[:, :, ::-1])[1].tofile('{}/{:0>9d}.png'.format(args.output,cnt))
        cnt += 1

files = [f for f in os.listdir(args.input)]
if not files:
    sys.exit(0)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=100)
_thread.start_new_thread(build_read_buffer, (args, read_buffer,files))
_thread.start_new_thread(clear_write_buffer, (write_buffer, None))
print('请忽略以上错误')
print('加载模型')
# set up model
# FIXME: currenly RRDBNet in BasicSR does not support scale argument. Will update later
# model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=args.scale)
torch.set_grad_enabled(False)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
loadnet = torch.load(args.model_path)
model.load_state_dict(loadnet['params_ema'], strict=True)
model.eval()
mod_scale = args.mod_scale

# if CUDA IS AVALIABLE
const_model_memory_usage = 0.6
const_pixel_memory_usage = 0.9 / 65536
total_memory = torch.cuda.get_device_properties(args.gpu_id).total_memory/(1024**3) * 0.8 * (args.precent/100)
available_memory = (total_memory - const_model_memory_usage)

pbar = tqdm(total=len(os.listdir(args.input)))
RRDB_instance = RRDBNetUpscaler(model_net=model, scale=mod_scale, device=device)
while True:
    img = read_buffer.get()
    if img is None:
        break
    H,W,C = img.shape
    tile_size = int(math.sqrt(available_memory / const_pixel_memory_usage)) * mod_scale
    padding = (mod_scale**2) / tile_size
    cl = esrlupscale.TiledUpscaler(upscaler=RRDB_instance, tile_size=tile_size, tile_padding=padding)
    write_buffer.put(cl.upscale(img))
    pbar.update(1)

print("等待文件写入，如果此步等待过长时间请直接退出：")
while not os.path.exists('{}/{:0>9d}.png'.format(args.output,cnt-1)):
    time.sleep(1)