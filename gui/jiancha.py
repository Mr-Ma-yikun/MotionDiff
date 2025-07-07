import torch
import matplotlib.pyplot as plt
import matplotlib
from torchvision import transforms
from colorwheel import flow_to_image
#from torchvision.utils import flow_to_image
import numpy as np
from PIL import Image
import h5py

#mask目前正常
mask = torch.load('mask.pth')
# with h5py.File('hammering_video9_frame0.h5', 'r') as f:
#     flow = f['flow'][:]
flow = torch.load('flow.pth',map_location='cpu')
#flow = flow.astype(np.float32)
mask_float = mask.float()

mask_squeezed = mask_float.squeeze()
averaged_mask = mask_squeezed.mean(dim=0, keepdim=True)

# 将通道数从 4 改为 3
averaged_mask_3_channels = averaged_mask.expand(3, -1, -1)
image = transforms.ToPILImage()(averaged_mask_3_channels)


image.save("mask.png")


#flow_numpy = flow.squeeze(0).detach().cpu().numpy()  # 将张量转换为 NumPy 数组，并移除第一个维度

#对论文的flow tensor(1,2,512,512)

#也就是说，步进需要得到这个，还需要将光流放到目标区域
flow_numpy = flow.squeeze().permute(1, 2, 0).cpu().numpy()
#flow_numpy = np.transpose(flow, (1, 2, 0))#.squeeze().permute(1, 2, 0).detach().numpy()
#flow_numpy = flow
print(flow_numpy.shape)
flow = flow_to_image(flow_numpy)#.squeeze(0)
flow = Image.fromarray(flow)
#image = transforms.ToPILImage()(flow)
flow.save("flow.png")


#flow_to_image_torch(flow)