import cv2
import numpy as np
import torch
import h5py
from colorwheel import flow_to_image

# 扩充mask区域  -----------假设 mask 是原始的 mask 区域，flow 是光流场
def expand_mask_with_flow(mask, flow, threshold=1.0):
    # 复制mask图像，保持原mask区域不变
    expanded_mask = mask.copy()
    # 遍历整个图
    # 根据光流扩充mask
    mask_points = np.where(mask == 255)  # 获取mask区域中所有的白色点的坐标

    for i in range(len(mask_points[0])):
        px = mask_points[1][i]  # x坐标
        py = mask_points[0][i]  # y坐标

        displacement = flow[py, px]  # 获取该点的光流位移信息
        new_px = int(px + displacement[0])  # 计算新的x坐标
        new_py = int(py + displacement[1])  # 计算新的y坐标
        new_px = np.clip(new_px, 0, flow.shape[1] - 1)  # 防止越界
        new_py = np.clip(new_py, 0, flow.shape[0] - 1)  # 防止越界

        # 计算扩充采样点
        start = (px, py)
        end = (new_px, new_py)
        distance = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        num_samples = int(distance)  # 可以根据需要对值进行调整
        # 使用 np.linspace 生成直线上的所有像素点
        line_points_x = np.linspace(start[0], end[0], num=num_samples).astype(int)
        line_points_y = np.linspace(start[1], end[1], num=num_samples).astype(int)

        # 将直线上的点设为 mask
        for x, y in zip(line_points_x, line_points_y):
            expanded_mask[y, x] = 255  # 将直线上的点设为白色，表示 mask 扩展后的区域
    expanded_mask = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR)
    return expanded_mask

#选择不同的模式，采用不同的warp和mask策略
modes = ['pingyi', 'xuanzhuan', 'suoxiao', 'fangda', 'lashen']
mode = modes[0]
image = cv2.imread('280.jpg')
flow = torch.load('flow.pth')[0].numpy()

#2, 512, 512

flow_color = flow_to_image(np.transpose(flow, [1,2,0]))
cv2.imwrite('flow_color.png', flow_color)

#*---------------------------------------------------第一阶段，得到运动的粗略图像
h,w,c = image.shape
flow_coor = np.array(flow, dtype=np.int64)

#这个没问题
flow_X = flow_coor[0]
flow_Y = flow_coor[1]
base_coor = np.mgrid[0:h, 0:w]

#纯白图像
#
coarse_warp = np.zeros_like(image) + 255
#运动的坐标
move_coor = base_coor + np.array([flow_Y, flow_X])

coor = np.where((base_coor[1] != move_coor[1]) + (base_coor[0] != move_coor[0]))
move_coor = np.transpose(move_coor, [1,2,0])

#运动的坐标值

coarse_warp[(np.clip(move_coor[coor][:, 0], 0, 511), np.clip(move_coor[coor][:, 1], 0, 511))] = image[coor]
#coarse_warp[(move_coor[coor][:, 0], move_coor[coor][:, 1])] = image[coor]

coarse_warp = np.array(coarse_warp, dtype='uint8')

# Save the result
cv2.imwrite('coarse_warp.png', coarse_warp)

#-----------------------------------------------------------------第二阶段，计算整体的填充区域

x = flow[0]
y = flow[1]
mask = np.zeros_like(x)
mask[(x != 0) & (y != 0)] = 255
flow_np = np.transpose(flow, [1, 2, 0])

#对于放大和拉伸的离散情况，需要进行区域扩充
if mode == 'fangda': #or mode == 'lashen':

    expand_mask = expand_mask_with_flow(mask, flow_np, threshold=1.0)
    gray_image = cv2.cvtColor(expand_mask, cv2.COLOR_BGR2GRAY)

    # 应用中值滤波去除噪声
    median_filtered_image = cv2.medianBlur(gray_image, 3)  # 中值滤波，内核大小为5x5

    # 定义形态学闭运算的结构元素
    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 形态学闭运算填充小孔
    expand_mask_2 = cv2.morphologyEx(median_filtered_image, cv2.MORPH_CLOSE, kernel)
    expand_mask = np.expand_dims(expand_mask_2, axis=-1)

# 对于缩小、平移旋转等密集情况，只使用warp后的mask
else:

    expand_mask = (coarse_warp != 255).astype(np.uint8) * 255

    if mode == 'xuanzhuan' or mode == 'suoxiao' or mode =='lashen':
        # 将图像转换为灰度图
        gray_image = cv2.cvtColor(expand_mask, cv2.COLOR_BGR2GRAY)

        # 应用中值滤波去除噪声
        median_filtered_image = cv2.medianBlur(gray_image, 5)  # 中值滤波，内核大小为5x5

        # 定义形态学闭运算的结构元素
        kernel_size = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 形态学闭运算填充小孔
        expand_mask_2 = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        expand_mask = np.expand_dims(expand_mask_2, axis=-1)

cv2.imwrite('mask.png', expand_mask)

#-----------------------------------------------------------------第三阶段，根据mask区域插值，得到最终的运动物体

import cv2
import numpy as np
from scipy.spatial import cKDTree

im1 = coarse_warp
mask = expand_mask[:,:,0]

index = mask != 0
index_ori = np.mean(im1, axis=2) != 255
inter_index = index * ~index_ori
query_index = index_ori
coor_inter = np.array(np.where(inter_index)).T
coor_query = np.array(np.where(query_index)).T
tree = cKDTree(coor_query)
distance, indices = tree.query(coor_inter)
nearest_coordinates = coor_query[indices]
im1[(coor_inter[:,0],coor_inter[:,1])] = im1[(nearest_coordinates[:,0],nearest_coordinates[:,1])]
cv2.imwrite('warped_fine.png', im1)

mask = expand_mask/255.0
next_frame = mask * im1 + (1- mask) * image
cv2.imwrite('next_frame.png', next_frame)

