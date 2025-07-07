import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from flow_viz import flow_to_image
from PIL import Image
from flow_utils import RAFT
import numpy as np
import torch.nn.functional as F

def optical_flow_warping(x, flo, pad_mode, device):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo  # warp后，新图每个像素对应原图的位置

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=pad_mode)

    mask = torch.ones(x.size()).to(device)
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


#warp的物体mask值为1，背景区域为0
class FlowLoss(nn.Module):
    def __init__(self, color_weight=100.0, 
                 flow_weight=1.0):
        
        super().__init__()

        self.flow_net = RAFT()

        self.flow_weight = flow_weight
        self.color_weight = color_weight


    def masked_l1(self, x, y, mask):
        mask = mask.to(x.device)
        x = x * mask
        y = y * mask
        return l1_loss(x, y)

    #                生成图像， 原始图像，目标光流
    def forward(self, pre_img, src_img, gt_flow, device):

        self.target_flow = gt_flow

        # Normalize
        target_img = src_img.clone() / 2. + 0.5
        pred_img = pre_img.clone() / 2. + 0.5
        
        # RAFT估计光流
        flow_est = self.flow_net(target_img, pred_img)

        #光流Loss保证背景区域没有奇怪的形状
        flow_gt = self.target_flow.to(device)
        flow_loss = l1_loss(flow_gt, flow_est)

        #将生成的图，warp到原始形状，然后计算全局Loss
        
        pre_warp = optical_flow_warping(pred_img, flow_gt,pad_mode='border', device=device)
        color_loss = l1_loss(pre_warp, target_img)
        loss_total = self.flow_weight * flow_loss + self.color_weight * color_loss

        # Make info
        flow_im = Image.fromarray(flow_to_image(flow_est[0].permute(1,2,0).cpu().detach().numpy()))

        info = {}
        info['flow_loss'] = flow_loss.item()
        info['color_loss'] = color_loss.item()
        info['flow_im'] = flow_im

        return loss_total, info

