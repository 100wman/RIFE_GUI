import cv2
import numpy as np

def mean_scale(HR,targetHeight,targetWidth):
    h,w,c = HR.shape
    BChannel = HR[:,:,0]
    GChannel = HR[:,:,1]
    RChannel = HR[:,:,2]
    ystep = h / targetHeight
    xstep = w / targetWidth
    map = np.zeros((targetHeight,targetWidth,c),np.float32)
    for y in range(targetHeight-1):
        for x in range(targetWidth-1):
            B = BChannel[int(y*ystep):int(ystep*(y+1))-1,int(x*xstep):int(xstep*(x+1))-1].mean()
            G = GChannel[int(y*ystep):int(ystep*(y+1))-1,int(x*xstep):int(xstep*(x+1))-1].mean()
            R = RChannel[int(y*ystep):int(ystep*(y+1))-1,int(x*xstep):int(xstep*(x+1))-1].mean()
            map[y,x] = [B,G,R]
    return map.astype(np.uint8)

def get_scale(scale_list,i0,i1):
    disssim = (1 - compare_ssim(i0,i1,multichannel=True)) * 100
    max = len(scale_list)
    if disssim > max:
        return scale_list[max-1]
    else:
        return scale_list[int(disssim)]

i0 = cv2.imread()
i1 = cv2.imread()
i0 = mean_scale(i0,8,8)
i1 = mean_scale(i1,8,8)
scale_list = [1.0,0.5,0.25]
print(get_scale(scale_list,i0,i1))