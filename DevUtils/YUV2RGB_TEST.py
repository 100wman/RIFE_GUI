# encoding=utf-8
#  reader: {'-vsync': '0', '-hwaccel': 'auto'} {'-map': '0:v:0', '-vframes': '10000000000', '-sws_flags': 'lanczos', '-color_range': 'tv', '-color_primaries': 'bt2020', '-colorspace': 'bt2020nc', '-color_trc': 'arib-std-b67', '-s': '960x540', '-vf': 'copy,minterpolate=fps=50.000:mi_mode=dup'}
import cv2
import numpy as np
import torch

from skvideo.io import FFmpegReader

# return FFmpegReader(filename=self.ARGS.input, inputdict=input_dict, outputdict=output_dict)
inputdict = {'-vsync': '0', }
# outputdict = {'-map': '0:v:0', '-vframes': '10000000000', '-sws_flags': 'lanczos', '-color_range': 'tv', '-color_primaries': 'bt2020', '-colorspace': 'bt2020nc', '-color_trc': 'arib-std-b67', '-s': '960x540', '-vf': 'copy,minterpolate=fps=50.000:mi_mode=dup'}
outputdict = {'-map': '0:v:0', '-vframes': '10000000000', '-color_range': 'tv', '-color_primaries': 'bt2020',
              '-colorspace': 'bt2020nc', '-color_trc': 'arib-std-b67',
              '-vf': 'copy,minterpolate=fps=25.000:mi_mode=dup', '-c:v': 'v410', '-pix_fmt': 'yuv444p16le'}
filename = r"D:\60-fps-Project\input or ref\Test\【1】1080p基础测试.mp4"
filename = r"D:\60-fps-Project\input or ref\Test\【3】HDR10测试.mkv"
reader = FFmpegReader(filename=filename, inputdict=inputdict, outputdict=outputdict).nextFrame()
window = 'frame'

"""
yuv444 ---bt601--> rgb
[[0.00455872,   	0.00000000,   	0.00624866 ],  
[0.00455872,   	-0.00153632,   	-0.00319838],   
[0.00455872,   	0.00791071,   	-0.00001027]]   

yuv444 ---bt709--> rgb
[[0.00456634,   	-0.00000003,   	0.00703030],
[0.00456626,   	-0.00083628,   	-0.00208986],   
[0.00456709,   	0.00828377,   	-0.00000040]]   


yuv444 ---bt2020--> rgb
[[0.00455872,   	0.00000000,   	0.00624866 ],  
[0.00455872,   	-0.00153632,   	-0.00319838],   
[0.00455872,   	0.00791071,   	-0.00001027]]   


"""
device = torch.device('cuda')
bitdepth = 16
w, h = 3840, 2160
yuv_rgb_bit_conv = 2 ** (bitdepth - 8)
sub_matrix = np.array([16., 128., 128.])


def get_torch_matrix(transfer_matrix):
    transfer_matrix = torch.from_numpy(transfer_matrix).to(device, non_blocking=True).float().repeat(h, w, 1,
                                                                                                     1)  # h,w,3,3
    return transfer_matrix


bt709_matrix = np.array([[0.00456634, -0.00000003, 0.00703030],
                         [0.00456626, -0.00083628, -0.00208986],
                         [0.00456709, 0.00828377, -0.00000040]])
bt709_matrix = np.array([[0.00455872,   	0.00000000,   	0.00624866 ],
                        [0.00455872,   	-0.00153632,   	-0.00319838],
                        [0.00455872,   	0.00791071,   	-0.00001027]]   )

bt709_matrix = np.expand_dims(np.expand_dims(bt709_matrix, 0), 0).repeat(h, 0).repeat(w, 1)


# bt709_matrix = get_torch_matrix(bt709_matrix)


def yuv2bt709rgb(yuv_frame: np.ndarray):
    yuv_frame = yuv_frame / yuv_rgb_bit_conv
    yuv_frame = yuv_frame - sub_matrix
    yuv_frame = np.expand_dims(yuv_frame, 3)
    # yuv_frame = get_torch_matrix(yuv_frame)
    rgb_frame = bt709_matrix @ yuv_frame
    rgb_frame = (rgb_frame.squeeze().clip(0, 1) * 255.).astype(np.uint8)
    # rgb_frame = (rgb_frame * 255.).squeeze(3).byte().cpu().numpy().copy()
    return rgb_frame


for frame in reader:
    frame = yuv2bt709rgb(frame)
    print(frame.shape)
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow(window, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
cv2.destroyAllWindows()
# break
