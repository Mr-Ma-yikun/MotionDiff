from colorwheel import flow_to_image
from PIL import Image
import numpy as np
import cv2
import torch

def Save(flow):

    # 保存最终结果
    result_tensor = torch.from_numpy(flow).unsqueeze(0).permute(0, 3, 1, 2)  # 假设数据存储顺序需要修改
    # 保存张量到 .pth 文件
    #(1,2,512,512) tensor
    result_tensor = result_tensor.float()
    torch.save(result_tensor, 'flow.pth')

def get_translation(x1, y1, x2, y2):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates
    flow = np.zeros((512, 512, 2))
    flow[:, :, 0] = x2 - x1
    flow[:, :, 1] = y2 - y1

    # Get image and paste to background
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow


def get_rotation(x1, y1, x2, y2):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates
    #theta = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 400 * np.pi

    theta = np.arctan2(y2 - y1, x2 - x1)
    coords = np.arange(0, 512)
    xx, yy = np.meshgrid(coords, coords)
    xx = xx - x1
    yy = yy - y1
    dx = np.cos(theta) * xx - np.sin(theta) * yy
    dy = np.sin(theta) * xx + np.cos(theta) * yy

    flow = np.zeros((512, 512, 2))
    flow[:, :, 0] = dx #* 0.2
    flow[:, :, 1] = dy #* 0.25

    # Get image and paste to background
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    print('hudu:',theta)
    print('jiaodu:',np.degrees(theta))
    return flow_im, flow


def get_scale(x1, y1, x2, y2):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates
    factor = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 100
    coords = np.arange(0, 512)
    xx, yy = np.meshgrid(coords, coords)
    xx = xx - x1
    yy = yy - y1

    flow = np.zeros((512, 512, 2))
    flow[:, :, 0] = xx * (factor - 1)
    flow[:, :, 1] = yy * (factor - 1)

    # Get image and paste to background
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow


def get_scale_1d(x1, y1, x2, y2):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates
    factor = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 100
    theta = np.arctan2(x2 - x1, y2 - y1)
    coords = np.arange(0, 512)
    u = np.array([y2 - y1, -x2 + x1])
    #这个网格是以x1 y1为中心的
    xx, yy = np.meshgrid(coords, coords)
    xx = xx - x1
    yy = yy - y1
    v = np.stack((xx, yy), axis=2)

    o = v - ((v @ u) / (u @ u))[:, :, None] * u
    flow = o * (factor - 1)
    print('x1,y1:', x1, y1)
    print('x2,y2:', x2, y2)
    # #保证不在同一点
    # if x1 != x2 and y1!= y2:
    #     # ----------计算垂线，理论上该垂线即为光流分割面/线
    #     m_original = (y2 - y1) / (x2 - x1)
    #     perpendicular_line_flow_values = []
    #     #垂线斜率
    #     m_perpendicular = -1 / m_original
    #     A = m_perpendicular
    #     B = -1
    #     C = y1 - m_perpendicular * x1
    #     x_range = np.linspace(min(x1, x2) - 100, max(x1, x2) + 100, 200)  # 你可以根据需要调整这个范围
    #     y_range = (-A * x_range - C) / B
    #
    #     # 对于垂线上的每个点，找到其最接近的整数坐标，并获取光流值
    #     for x, y in zip(x_range, y_range):
    #         # 找到最接近的整数坐标（因为光流数据的坐标是整数）
    #         x_int = int(round(x))
    #         y_int = int(round(y))
    #
    #         # 确保坐标在光流数据范围内
    #         if 0 <= x_int < 512 and 0 <= y_int < 512:
    #             # 获取该点的光流值
    #             u_value = flow[y_int, x_int ,0]
    #             v_value = flow[y_int, x_int, 1]
    #             print(u_value, v_value)
    #             # 存储光流值（可以根据需要存储u, v或模长）
    #             perpendicular_line_flow_values.append((u_value, v_value))
    #
    #     # Get image and paste to background
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow


def get_bezier(x0, control_points):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates

    # Sort by y-value
    control_points = sorted(control_points, key=lambda x: x[1])

    amps = np.array(control_points)[:, 0] / 100 - 1
    ys = np.array(control_points)[:, 1]

    xx = np.arange(0, 512)
    xx = xx - x0

    flow = np.zeros((512, 512, 2))

    flow[:ys[0], :, 0] = xx * amps[0]
    flow[ys[-1]:, :, 0] = xx * amps[-1]

    if len(control_points) >= 2:
        for i in range(len(control_points) - 1):
            for y in range(ys[i], ys[i + 1]):
                t = (y - ys[i]) / (ys[i + 1] - ys[i])  # btwn [0,1]
                factor = (amps[i + 1] - amps[i]) * (1 - np.cos(t * 3.141592)) / 2. + amps[i]
                flow[y, :, 0] = xx * factor

    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow