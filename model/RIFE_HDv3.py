from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from model.IFNet_HDv3 import *
from model.loss import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, use_multi_cards=False, forward_ensemble=False, tta=False, local_rank=-1):
        self.tta = tta
        self.forward_ensemble = forward_ensemble
        self.use_multi_cards = use_multi_cards
        self.device_count = torch.cuda.device_count()
        if self.device_count > 1 and self.use_multi_cards:
            self.flownet = nn.DataParallel(IFNet())
        else:
            self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location='cpu')),
                                             False)

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        merged = self.flownet(imgs, scale_list, ensemble=self.forward_ensemble)[2]
        if not self.tta:
            return merged
        else:
            imgs_l1 = torch.cat((img0, merged), 1)
            imgs_r1 = torch.cat((merged, img1), 1)
            img_025 = self.flownet(imgs_l1, scale_list, ensemble=self.forward_ensemble)[2]
            img_075 = self.flownet(imgs_r1, scale_list, ensemble=self.forward_ensemble)[2]
            imgs_con = torch.cat((img_025, img_075), 1)
            return self.flownet(imgs_con, scale_list, ensemble=self.forward_ensemble)[2]

if __name__ == '__main__':
    _img0 = torch.zeros(1, 3, 256, 256).float().to(device)
    _img1 = torch.tensor(np.random.normal(
        0, 1, (1, 3, 256, 256))).float().to(device)
    _imgs = torch.cat((_img0, _img1), 1)
    model = Model(True, True, True)
    model.eval()
    print(model.inference(_img0, _img1).shape)
