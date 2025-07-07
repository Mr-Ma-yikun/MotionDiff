import os
import shutil

import torch
import numpy as np
import math
import pygame
from PIL import Image
#from torchvision.utils import flow_to_image
from torchvision import transforms
from colorwheel import flow_to_image
from scipy.ndimage import label, find_objects

import cv2
from get_image import Save

MODES = ['translate', 'rotate', 'scale', 'scale_1d']
MODE = MODES[0]
im_path = '280.jpg'
im = Image.open(im_path)
im = np.array(im)[:,:,:3]
shade = 0.5

### START SAM STUFF ###
from segment_anything import SamPredictor, sam_model_registry

sam_checkpoint = r"path/of/sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(im)
### END SAM STUFF ###


def expand_mask_with_flow(mask, flow, threshold=1.0):
    #复制mask图像，保持原mask区域不变
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

        expanded_mask[new_py, new_px] = 255  # 将直线上的点设为白色，表示 mask 扩展后的区域

    expanded_mask = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR)
    return expanded_mask


def find_flow_region_boundaries(flow):
    # 找到非零像素的坐标
    non_zero_coords = np.argwhere(flow != 0)

    # 找到最左、最右、最上和最下的像素位置
    #(y,x)
    leftmost_coord = non_zero_coords[np.argmin(non_zero_coords[:, 1])]
    rightmost_coord = non_zero_coords[np.argmax(non_zero_coords[:, 1])]
    topmost_coord = non_zero_coords[np.argmin(non_zero_coords[:, 0])]
    bottommost_coord = non_zero_coords[np.argmax(non_zero_coords[:, 0])]

    return leftmost_coord[:2], rightmost_coord[:2], topmost_coord[:2], bottommost_coord[:2]


def calculate_new_coordinates(flow):
    new_coordinates = []
    height, width, _ = flow.shape
    for i in range(height):
        for j in range(width):
            #计算所有光流区域的坐标点移动
            if flow[i, j, 0]!=0 and flow[i, j, 1]!=0:
                new_x = j + flow[i, j, 0]  # 计算新的 x 坐标
                new_y = i + flow[i, j, 1]  # 计算新的 y 坐标
                if new_x != 0 or new_y != 0:
                    new_coordinates.append([int(new_y), int(new_x)])

    return new_coordinates


if MODE == 'translate':
    from get_image import get_translation as get_image
elif MODE == 'rotate':
    from get_image import get_rotation as get_image
elif MODE == 'scale':
    from get_image import get_scale as get_image
elif MODE == 'scale_1d':
    from get_image import get_scale_1d as get_image

def show_image(image):
    image_surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
    screen.blit(image_surface, (0, 0))
    pygame.display.flip()

# Initialize Pygame
pygame.init()

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Set up the screen
screen_width, screen_height = 512, 512
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Arrow App")
screen.fill(white)

# Draw image
image = Image.open(im_path)
show_image(image)

# Arrow properties
arrow_start = None
arrow_end = None
drawing = False

def draw_arrow(surface, color, start, end, width):
    arrow_size = 7
    pygame.draw.line(surface, color, start, end, width)
    theta = math.atan2((start[0]-end[0]), (start[1]-end[1])) + math.pi / 3
    pygame.draw.polygon(surface, color, (
                  (end[0]+arrow_size*math.sin(theta), 
                   end[1]+arrow_size*math.cos(theta)), 
                  (end[0]+arrow_size*math.sin(theta+2*math.pi/3), 
                   end[1]+arrow_size*math.cos(theta+2*math.pi/3)), 
                  (end[0]+arrow_size*math.sin(theta-2*math.pi/3), 
                   end[1]+arrow_size*math.cos(theta-2*math.pi/3))))

# Main game loop
running = True
getting_mask = True
darwing = None
while running:
    for event in pygame.event.get():
        if getting_mask:
            #鼠标按下开始画框
            if event.type == pygame.MOUSEBUTTONDOWN:
                darwing = True
                #起始坐标
                start_pos = event.pos
                current_rect = (start_pos, start_pos)  # 初始化为一个点

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False

                end_pos = event.pos
                x1, y1 = min(start_pos[0], end_pos[0]), min(start_pos[1], end_pos[1])
                x2, y2 = max(start_pos[0], end_pos[0]), max(start_pos[1], end_pos[1])

                input_box = np.array([x1, y1, x2, y2])


                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

                mask = masks[scores.argmax()]

                #(512,512,3)
                mask = np.stack([mask]*3,axis=2).astype(float)

                g = 0.6
                bg = Image.fromarray((im * ((1-g) + g * mask.astype(float))).astype(np.uint8))

                bg = (shade * im * (1 - mask) + 255 * mask).astype(np.uint8)
                show_image(Image.fromarray(bg))

                getting_mask = False

            elif event.type == pygame.MOUSEMOTION and darwing:
                current_rect = (start_pos, event.pos)

                rect_pos = (min(current_rect[0][0], current_rect[1][0]), min(current_rect[0][1], current_rect[1][1]),
                            abs(current_rect[1][0] - current_rect[0][0]), abs(current_rect[1][1] - current_rect[0][1]))

                #更新图片，画框，显示，再更新图片
                show_image(image)
                pygame.draw.rect(screen, (0,255,0), rect_pos)
                    # 更新屏幕显示
                pygame.display.flip()
        else:
            if event.type == pygame.QUIT:
                running = False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_RETURN:

                    mask_uint8 = (mask * 255).astype(np.uint8)

                    gray_mask = cv2.cvtColor(mask_uint8, cv2.COLOR_BGR2GRAY)

                    clipped_flow = np.zeros((512, 512, 2))
                    clipped_flow[gray_mask == 255] = flow[gray_mask == 255]
                    Save(clipped_flow)

                    new_mask = expand_mask_with_flow(gray_mask, clipped_flow, 1.0)

                    kernel = np.ones((5, 5), np.uint8)
                    # 对 inverted_mask 进行膨胀操作
                    downsampled_image = cv2.dilate(new_mask, kernel, iterations=3)

                    downsampled_image = cv2.resize(downsampled_image, (64, 64))

                    downsampled_image = 255 - downsampled_image  # 翻转颜色

                    black_image = np.zeros_like(downsampled_image)+255

                    # 遍历每个像素，如果它不是白色（即RGB值不全为255），则将其设为黑色
                    for i in range(downsampled_image.shape[0]):
                        for j in range(downsampled_image.shape[1]):
                            if not np.all(downsampled_image[i, j] == 255):
                                black_image[i, j] = [0, 0, 0]

                    with open('color_pixels.txt', 'w') as f:
                        # 遍历图像的每个像素
                        for row in black_image:
                            for pixel in row:
                                # 写入BGR三个通道的值，用空格分隔
                                f.write(' '.join(map(str, pixel)) + '\n')
                    tensor_image = torch.from_numpy(black_image)
                    transposed_tensor = tensor_image.permute(2, 0, 1)  # (3,64,64)
                    new_channel = transposed_tensor[2, :, :].unsqueeze(0)  # 获取第三个通道的值并添加一个维度
                    new_tensor = torch.cat([transposed_tensor, new_channel], dim=0)
                    new_tensor = new_tensor.unsqueeze(0)

                    bool_torch_image = new_tensor.bool()
                    torch.save(bool_torch_image, 'mask.pth')
                    running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                arrow_start = pygame.mouse.get_pos()
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                # Clear screen
                screen.fill(white)

                arrow_end = pygame.mouse.get_pos()

                # Calculate dx and dy from arrow_start to arrow_end
                dx = arrow_end[0] - arrow_start[0]
                dy = arrow_end[1] - arrow_start[1]

                # Get the image based on dx and dy
                flow_image, flow = get_image(arrow_start[0], arrow_start[1], arrow_end[0], arrow_end[1])


                # Mask Image
                image = (shade * im * (1 - mask) + flow_image * mask).astype(np.uint8)
                image = Image.fromarray(image)
                # Convert PIL Image to Pygame surface
                image_surface = pygame.image.fromstring(image.tobytes(), image.size, image.mode)
                # Draw the new background
                screen.blit(image_surface, (0, 0))

                # Draw arrow while dragging
                if MODE == 'translate':
                    draw_arrow(screen, black, arrow_start, pygame.mouse.get_pos(), 3)
                elif MODE == 'rotate':
                    rotation_end = (
                            int(arrow_end[0] + dy / 2.),
                            int(arrow_end[1] - dx / 2.)
                                )
                    draw_arrow(screen, black, arrow_end, rotation_end, 3)
                    pygame.draw.circle(screen, black, arrow_start, 10, width=3)
                elif MODE == 'scale':
                    dr = math.sqrt(dx**2+dy**2)
                    start = (arrow_start[0] + int(dx / dr * 100), arrow_start[1] + int(dy / dr * 100))
                    draw_arrow(screen, black, start, pygame.mouse.get_pos(), 3)
                    pygame.draw.circle(screen, black, arrow_start, 100, width=3)
                elif MODE == 'scale_1d':
                    draw_arrow(screen, black, arrow_start, pygame.mouse.get_pos(), 3)

                    dr = math.sqrt(dx**2+dy**2)
                    ldx = -int(dy / dr * 10)
                    ldy = int(dx / dr * 10)
                    udx = int(dx / dr * 100)
                    udy = int(dy / dr * 100)
                    start = (arrow_start[0] + ldx, arrow_start[1] + ldy)
                    end = (arrow_start[0] - ldx, arrow_start[1] - ldy)
                    pygame.draw.line(screen, black, start, end, 3)

                    start = (arrow_start[0] + udx + ldx, arrow_start[1] + udy + ldy)
                    end = (arrow_start[0] + udx - ldx, arrow_start[1] + udy - ldy)
                    pygame.draw.line(screen, black, start, end, 3)

                pygame.display.flip()


# Quit Pygame
pygame.quit()

