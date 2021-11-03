import cv2
import numpy as np
import torch

#跳转到38行开始看
def to_tensor(*imgs):
    if TenHalf:
        return [torch.from_numpy(np.transpose(img, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).half() / 255. for img in imgs]
    return [torch.from_numpy(np.transpose(img, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255. for img in imgs]

ensemble = True
tta = False
TTIter = 1
TTM = 'side_vector'
TenHalf = False
model_path = r"D:\60-fps-Project\Projects\RIFE GUI\train_log\official_2.3"

device = torch.device("cuda")
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if TenHalf:
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
try:
    from RIFE.RIFE_HDv3 import Model
    model = Model()
    model.load_model(model_path,-1)
    version = 3
except:
    from RIFE.RIFE_HDv2 import Model
    model = Model()
    model.load_model(model_path,-1)
    version = 2
model.eval()
model.device()
spent = 0

#新padding方法,可以换旧的
def get_padding(img,scale):
    h,w,_ = img.shape
    tmp = max(32, int(32 / scale))
    right = ((w - 1) // tmp + 1) * tmp - w
    bottom = ((h - 1) // tmp + 1) * tmp - h
    return int(right),int(bottom)

def make_inference(im0, im1, exp, scale ,tta):   
    #加黑边padding
    right,bottom = get_padding(im0,scale)
    t0 = cv2.copyMakeBorder(im0,0,bottom,0,right,cv2.BORDER_CONSTANT)
    t1 = cv2.copyMakeBorder(im1,0,bottom,0,right,cv2.BORDER_CONSTANT)
    #转换为tensor
    I0,I1 = to_tensor(t0,t1)
    global model
    if version == 2:
        middle = model.inference(I0, I1, scale, TTIter)
    elif version == 3:
        middle = model.inference(I0, I1, scale, TTIter)
    I0 = I1 = 0
    mid = (((middle[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
    if exp == 1:
        return [mid]
    first_half = make_inference(im0, mid, exp=exp - 1, scale=scale, tta=tta)
    second_half = make_inference(mid, im1, exp=exp - 1, scale=scale, tta=tta)
    return [*first_half, mid, *second_half]

i0 = cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\test\images\0.png")
i1 = cv2.imread(r"D:\60-fps-Project\Projects\RIFE GUI\test\images\1.png")
scale = 1.0 #开spilt后不建议调，不过可以复合用

# 建议:用户不指定就不开，1080P spilt开2，4K开4，8K开8，依此类推
spilt = 2 # spilt值为x,图像分裂为x**2块

h,w,c = i0.shape #高度，宽度，通道数
ori_h,ori_w,_ = [h,w,c] #原高度，原宽度

#图像将被处理为同时满足模型要求和算法要求的形式（加黑边
tmp = max(32, int(32 / scale))
while h % spilt != 0 or h % tmp != 0:
    h += 1
while w % spilt != 0 or w % tmp != 0:
    w += 1
i0 = cv2.copyMakeBorder(i0,0,h-ori_h,0,w-ori_w,borderType=cv2.BORDER_CONSTANT) #加黑边，padding
i1 = cv2.copyMakeBorder(i1,0,h-ori_h,0,w-ori_w,borderType=cv2.BORDER_CONSTANT)

# 生成下采样mask，用法 x = img[mask]
def gen_mask(iterY,iterX,h,w,c,spilt=4):
    LineY = np.linspace(iterY,h+iterY,int(h/spilt),False,dtype=np.int64) #生成一条y线段 iter -> (h - iter) 步长为 h / (int(h/y_spilt)-iter)
    LengthY = len(LineY)
    LineX = np.linspace(iterX,w+iterX,int(w/spilt),False,dtype=np.int64) #生成一条x线段 iter -> (w - iter) 步长为 w / (int(w/x_spilt)-iter)

    Cx = np.repeat(LineX,LengthY) #横坐标集
    Cy = np.array([],np.int64) #纵坐标集
    Cc = np.tile(np.array([ci for ci in range(c)],dtype=np.int64),len(Cx)) #通道集

    # n个通道重复N遍
    Cy = [np.concatenate((Cy,LineY)) for _ in range(int(len(Cx)/LengthY))]
    Cy = np.repeat(Cy,c)
    Cx = np.repeat(Cx,c)

    return (Cy,Cx,Cc)

ph,pw,pc = [int(h/spilt),int(w/spilt),c] #小图高度，宽度，通道数
mask_low = gen_mask(0,0,ph,pw,pc,1) #小图mask
mask_list = [] #大图的mask列表，用于还原大尺寸
pieces = [] #小图

for ys in range(spilt):
    for xs in range(spilt):
        mask = gen_mask(ys,xs,h,w,c,spilt) # 以ys,xs为起点，生成大图mask
        t0 = np.zeros((ph,pw,c),np.uint8)
        t1 = np.zeros((ph,pw,c),np.uint8)
        t0[mask_low] = i0[mask]
        t1[mask_low] = i1[mask]
        piece = make_inference(t0,t1,1,scale,tta)[0] #补帧出来的小图
        pieces.append(piece)
        mask_list.append(mask)

#合并图像
index = 0
map = np.zeros_like(i0) #创建大图地图
for ys in range(spilt):
    for xs in range(spilt):
        mask = mask_list[index]
        map[mask] = pieces[index][mask_low]
        cv2.imwrite(r"D:\60-fps-Project\Projects\RIFE GUI\test\images\p{}.png".format(index),pieces[index])
        index += 1

#自适应平滑 (去除棋盘格
kernel = np.ones((spilt,spilt),np.float32) / (spilt**2)
map = cv2.filter2D(map, -1, kernel) #最终结果

cv2.imwrite(r"D:\60-fps-Project\Projects\RIFE GUI\test\images\0-mid.png".format(index),map[0:ori_h,0:ori_w])
