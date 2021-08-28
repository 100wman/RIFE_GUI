

def inference(self, img0, img1, scale=1.0, ensemble=False, TTA=False):
    imgs = torch.cat((img0, img1), 1)
    scale_list = [4 / scale, 2 / scale, 1 / scale]
    merged = self.flownet(imgs, scale_list, ensemble=ensemble)[2]
    if TTA == False:
        return merged
    else:
        imgs_l1 = torch.cat((img0, merged), 1)
        imgs_r1 = torch.cat((merged, img1), 1)
        merged = None
        img_025 = self.flownet(imgs_l1, scale_list, ensemble=ensemble)[2]
        imgs_l1 = None
        img_075 = self.flownet(imgs_r1, scale_list, ensemble=ensemble)[2]
        imgs_r1 = None
        imgs_con = torch.cat((img_025, img_075), 1)
        img_025 = img_075 = None
        return self.flownet(imgs_con, scale_list, ensemble=ensemble)[2]

def inference(self, img0, img1, scale=1.0,ensembel = False,TTA=False):
    imgs = torch.cat((img0, img1), 1)
    flow, _ = self.flownet(imgs, scale)
    if ensembel:
        reimgs = torch.cat((img1, img0), 1)
        reflow,_ = self.flownet(reimgs,scale)
        flow = (flow + reflow) / 2
    merged = self.predict(imgs, flow, training=False)
    if TTA == False:
        return merged
    else:
        imgs_l1 = torch.cat((img0,merged),1)
        imgs_r1 = torch.cat((merged,img1),1)
        merged = None
        flow = self.flownet(imgs_l1, scale)
        img_025 = self.predict(imgs_l1, flow, training=False)
        imgs_l1 = None
        flow = self.flownet(imgs_r1, scale)
        img_075 = self.predict(imgs_r1, flow, training=False)
        imgs_r1 = None
        imgs_con = torch.cat((img_025,img_075),1)
        img_025 = img_075 = None
        flow = self.flownet(imgs_con, scale)
        return self.predict(imgs_con, flow, training=False)
