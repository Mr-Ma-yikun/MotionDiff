import torch
import numpy as np
import cv2
import torch.nn.functional as F
import open3d as o3d
import os

# data path
root_path = '/home/oseasy/mayikun/scannet/processed/'
scene_name = 'scene0008_00'

obj_id = 40
modes = ['pingyi', 'xuanzhuan', 'suoxiao', 'fangda', 'lashen']
mode = modes[4]

images = root_path + scene_name + '/color'
depths = root_path + scene_name + '/depth'
poses = root_path + scene_name + '/pose'
mesh_path = root_path + scene_name + '/' + scene_name + '_vh_clean_2.ply'
intrinsic = root_path + scene_name + '/intrinsic/intrinsic_depth.txt'

#-------------------------------------------------------选择要处理的图像id

image_ids = [640, 680, 700, 730]
used_index = 0
depth_map = []

for image_id in image_ids:
    image_path = os.path.join(images, f'{image_id}.jpg')
    depth_path = os.path.join(depths, f'{image_id}.png')

    image = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]  
   
    start_x = (width - 480) // 2  
    start_y = 0  # 高度不需要裁剪，所以从0开始
    cropped_image = image[start_y:start_y+480, start_x:start_x+480]  
    resized_image = cv2.resize(cropped_image, (512, 512))  

    cropped_depth = depth[start_y:start_y+480, start_x:start_x+480]  
    resized_depth = cv2.resize(cropped_depth, (512, 512), interpolation=cv2.INTER_LINEAR)

    image_save_path = os.path.join(root_path, scene_name, 'color_512', f'{image_id}.jpg')  
    depth_save_path = os.path.join(root_path, scene_name, 'depth_512', f'{image_id}.png')  

    image_save_dir = os.path.dirname(os.path.join(root_path, scene_name, 'color_512', f'{image_id}.jpg'))  
    if not os.path.exists(image_save_dir):  
        os.makedirs(image_save_dir)

    depth_save_dir = os.path.dirname(os.path.join(root_path, scene_name, 'depth_512', f'{image_id}.png'))  
    if not os.path.exists(depth_save_dir):  
        os.makedirs(depth_save_dir)

    cv2.imwrite(image_save_path, resized_image)  
    cv2.imwrite(depth_save_path, resized_depth)  
    depth_map.append(resized_depth)

#-------------------------------加载相机参数
K_ori = np.loadtxt(intrinsic)
fx_ori = K_ori[0,0]
fy_ori = K_ori[1,1]
cx_ori = K_ori[0,2]
cy_ori = K_ori[1,2]

#----------------------新的相机内参
scale_x = 512 / 480  
scale_y = 512 / 480   
fx = fx_ori * scale_x  
fy = fy_ori * scale_y
cx = 256
cy = 256

New_R = np.random.rand(len(image_ids), 3, 3)
New_T = np.random.rand(len(image_ids), 3)
New_ex = np.random.rand(len(image_ids), 4, 4)

for i, image_id in zip(range(len(image_ids)), image_ids):
    pose_path = os.path.join(poses, f'{image_id}.txt')
    
    with open(pose_path, 'r') as f:  
        pose_data = f.read()
        pose_matrix = np.fromstring(pose_data, sep=' ').reshape(4, 4)  
        New_ex[i]= pose_matrix
        New_R[i] = New_ex[i][:3, :3]
        New_T[i] = New_ex[i][:3, 3]

#------------------------------------------------mesh

meshes = o3d.io.read_triangle_mesh(mesh_path)

def vis_one_object(point_ids, scene_points):
    points = scene_points[point_ids]
    return point_ids, points

def get_obj_points(meshes, obj_id):

    label_colors, labels = [], []

    scene_points = np.asarray(meshes.vertices)

    pred = np.load(f'data/prediction/scannet_class_agnostic/{scene_name}.npz')

    masks = pred['pred_masks']
    num_instances = masks.shape[1]

    #建立label和point id的映射
    class_info = {}

    for idx in range(num_instances):
        mask = masks[:, idx]

        point_ids = np.where(mask)[0]

        point_ids, label_color = vis_one_object(point_ids, scene_points)

        label_colors.append(label_color)

        labels.append(str(idx))

        class_info[idx] = point_ids

    our_points_id = class_info[obj_id]
    our_points = scene_points[our_points_id]
    return our_points_id, our_points

our_points_id, our_points = get_obj_points(meshes, obj_id)

sub_mesh = meshes.select_by_index(our_points_id)  

points_dense_o3d = sub_mesh.sample_points_uniformly(512*512)
points_dense = np.asarray(points_dense_o3d.points)

o3d.io.write_triangle_mesh('obj.ply', sub_mesh)  

# center
points_dense_center = points_dense - np.mean(points_dense, axis=0)

#----------------------------------------------加载flow

#(1,2,512,512)

flow = torch.load('flow_lashen.pth',map_location='cpu')
flow_numpy = flow.squeeze().permute(1, 2, 0).cpu().numpy()

# ------------平移的深度计算模型，这个要优化下，目前只能水平？
def depth_with_pingyi(initial_depth,flow, pingyi_factor):

    updated_depth = initial_depth * (1 + flow * 500 * pingyi_factor)

    return updated_depth


# 扩充mask区域  -----------假设 mask 是原始的 mask 区域，flow 是光流场
def get_suofang_scale(flow, mode):

    if mode == 'suoxiao':
        #只使用正常光流区域
        nonzero_both = (flow[:, :, 0] != 0) & (flow[:, :, 1] != 0)

        # 计算每个像素点光流向量的模长
        magnitude = np.sqrt(np.square(flow[nonzero_both,0]) + np.square(flow[nonzero_both,1]))

        #尺度因子 = 平均运动 / 最大运动
        scale = np.mean(magnitude) / np.max(magnitude)
        return scale

    elif mode =='fangda':
        mask = np.zeros((512, 512), dtype=np.uint8)
        expanded_mask = mask.copy()

        # 找出光流中不为零的位置
        non_zero_indices = np.logical_and(flow[..., 0] != 0, flow[..., 1] != 0)

        # 将光流区域设为255
        mask[non_zero_indices] = 255

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

            #计算扩充采样点
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
        #expanded_mask = cv2.cvtColor(expanded_mask, cv2.COLOR_GRAY2BGR)

        ori = np.count_nonzero(mask == 255)
        extent = np.count_nonzero(expanded_mask == 255)
        #放大尺度因子 = 运动后的mask数量 / 原始mask区域数量

        scale = extent / ori

        return scale


# ------------缩放的点云 = 原始点云 * 缩放因子
def point_suofang(point_3d, scale_factor):

    New_3d = point_3d * scale_factor

    return New_3d

# -------------旋转稠密估计模型
def point_xuanzhuan(points, center, rot):
    rotated_points = []
    #对每个点都计算旋转
    for point in points:
        # 切换到旋转中心
        point_relative = np.array(point) - np.array(center)
        #旋转坐标
        rotated_point_relative = np.dot(rot, point_relative)
        #原始坐标系
        rotated_point = rotated_point_relative + np.array(center)
        #添加这个点
        rotated_points.append(rotated_point)

    return rotated_points

#--------------拉伸
# 定义一个函数来计算点到切面的距离
def point_to_plane_distance(point, A,B,C,D):
    """
    计算点到平面的距离。
    参数:
    point (np.ndarray): 点的坐标 (x, y, z)
    A,B,C,D
    返回:
    float: 点到平面的距离
    """

    #由于切面与Z平行，所以世界某点到该面的距离等于X2+Y2根号
    return (A * point[0] + B * point[1] + C * point[2] + D) / np.sqrt(A**2 + B**2 + C**2)

# 计算点云中每个点到切面的距离
def compute_distances_to_plane(point_cloud, A,B,C,D):
    """
    计算点云中每个点到切面的距离。
    参数:
    point_cloud (np.ndarray): 点云的坐标，形状为 (N, 3), 其中N是点的数量
    A,B,C,D为平面方程系数
    返回:
    np.ndarray: 形状为 (N,) 的数组，包含点云中每个点到切面的距离
    """
    distances = np.array([point_to_plane_distance(point, A,B,C,D) for point in point_cloud])
    return distances

def point_lashen(point_world_3d, x1, y1, x2, y2, flow, move_max_x, move_max_y):

    # ----------计算垂线，该垂线即为光流分割面/线
    #先要换到世界坐标系 (-1,1)
    x1 = (x1 * 2) / 512 - 1
    x2 = (x2 * 2) / 512 - 1
    y1 = (y1 * 2) / 512 - 1
    y2 = (y2 * 2) / 512 - 1

    m_original = (y2 - y1) / (x2 - x1)
    #垂线斜率
    m_perpendicular = -1 / m_original
    A = m_perpendicular
    B = -1
    C = y1 - m_perpendicular * x1
    x_range = np.linspace(min(x1, x2) - 50, max(x1, x2) + 50, 100)  # 你可以根据需要调整这个范围
    y_range = (-A * x_range - C) / B
    print(x_range[70], y_range[70])
    a = np.array([x1, y1])  # 点A的坐标，z坐标设为0因为线段在XY平面上

    # 交线为Ax + By + C =0
    # 平面方程为Ax + By + Cz + D = 0
    # 与XY垂直，所以简化为Ax + By + D = 0
    # 求解D值，带入交线点a
    D = a[1] - m_perpendicular * a[0]

    #计算正常光流的最大光流
    nonzero_both = (flow[:, :, 0] != 0) & (flow[:, :, 1] != 0)

    #最大值正常，例如100
    magnitude = np.sqrt(np.square(flow[nonzero_both, 0]) + np.square(flow[nonzero_both, 1]))
    max_magnitude = magnitude.max()
    distances = compute_distances_to_plane(point_world_3d,A,B,C,D)

    sc_fac = (distances / distances.max())

    old_x = point_world_3d[:,0]
    old_y = point_world_3d[:,1]

    d = point_world_3d[:,2]
    valid_indices = d != 0

    new_x = old_x.copy()
    new_y = old_y.copy()

    new_x[valid_indices] += sc_fac * move_max_x
    new_y[valid_indices] += sc_fac * move_max_y
    new_z = d.copy()

    New_point_world_3d = np.column_stack((new_x, new_y, new_z))

    return New_point_world_3d

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


def expand_mask_with_flow(mask, flow, threshold=1.0):
    leftmost_old, rightmost_old, topmost_old, bottommost_old = find_flow_region_boundaries(flow)
    # (y,x)
    new_coordinates = np.array(calculate_new_coordinates(flow))
    # 找到最小和最大值的索引

    # (x,y)
    leftmost_index = np.argmin(new_coordinates[:, 1])
    rightmost_index = np.argmax(new_coordinates[:, 1])
    topmost_index = np.argmin(new_coordinates[:, 0])
    bottommost_index = np.argmax(new_coordinates[:, 0])

    # 获取对应的坐标点
    leftmost_new = new_coordinates[leftmost_index]
    rightmost_new = new_coordinates[rightmost_index]
    topmost_new = new_coordinates[topmost_index]
    bottommost_new = new_coordinates[bottommost_index]
    all_points = [leftmost_old, rightmost_old, topmost_old, bottommost_old,
                  leftmost_new, rightmost_new, topmost_new, bottommost_new]

    # 按照 y 坐标排序
    all_points.sort(key=lambda point: point[1])

    # 水平边界点
    horizontal_left = all_points[0]
    horizontal_right = all_points[-1]

    # 按照 y 坐标排序
    all_points.sort(key=lambda point: point[0])

    # 垂直边界点
    vertical_top = all_points[0]
    vertical_bottom = all_points[-1]


    box_top_left = (horizontal_left[1], vertical_top[0])

    box_bottom_right = (horizontal_right[1], vertical_bottom[0])

    # 创建一个和原始矩阵相同大小的全零矩阵
    # 这里假设原始矩阵是 image
    box = np.zeros_like(mask)

    # 将框内的值设置为 255
    box[box_top_left[1]:box_bottom_right[1] + 1, box_top_left[0]:box_bottom_right[0] + 1] = 255
    expanded_mask = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
    return expanded_mask

#--------------------------------------------------读取数据集的内参矩阵

points_old = []
points_new = []

pingyi_offset = []
lashen_move_x = []
lashen_move_y = []

points_xuanzhuan_all = []
points_suofang_all = []
points_lashen_all = []

pcd_old = o3d.geometry.PointCloud()
pcd_new = o3d.geometry.PointCloud()

pingyi_scale = 0

suofang_scale = get_suofang_scale(flow_numpy, mode)
print('缩放因子:', suofang_scale)

points_id = image_ids

flows_views = np.zeros((len(image_ids),512,512,2), dtype=np.float32)


flows = flow_numpy

#遍历所有，得到点云
for v in range(512):
    for u in range(512):
        # 获取深度值
        d = depth_map[1][v, u]

        #保证是有效区域
        if d > 1.0:
            #旧点云和相机
            X_cam = (u - cx) * d / fx
            Y_cam = (v - cy) * d / fy
            Z_cam = d
            point_cam = np.array([X_cam, Y_cam, Z_cam, 1]).T
            # 使用外参将相机坐标系下的点转换到世界坐标系
            point_world = np.dot(New_ex[used_index], point_cam)
            point_world_3d = point_world[:3] / point_world[3]

            #如果没flow太小，不移动，保持原点云
            if flows[v, u][0] == 0 and flows[v, u][1] == 0:
                flow_at_uv = flows[v, u]
                New_X_cam = X_cam
                New_Y_cam = Y_cam
                New_Z_cam = d

            #如果是正常的运动，则进行运动计算
            else:
                #运动后的点云和相机
                flow_at_uv = flows[v, u]
                #判断是否在图像内
                new_u = u + flow_at_uv[0]
                new_v = v + flow_at_uv[1]
                New_X_cam = (new_u - cx) * d / fx
                New_Y_cam = (new_v - cy) * d / fy

            if mode == 'pingyi':
                #在计算平移时，记录稀疏视角偏移
                New_Z_cam = depth_with_pingyi(d, flow_at_uv[1], pingyi_scale)
                New_point_cam = np.array([New_X_cam/d * New_Z_cam, New_Y_cam/d * New_Z_cam, New_Z_cam, 1]).T
                New_point_world = np.dot(New_ex[used_index], New_point_cam)
                New_point_world_3d = New_point_world[:3] / New_point_world[3]
                pingyi_offset.append(New_point_world_3d - point_world_3d)

            elif mode == 'lashen':
                New_Z_cam = depth_with_pingyi(d, flow_at_uv[1], pingyi_scale)
                New_point_cam = np.array([New_X_cam / d * New_Z_cam, New_Y_cam / d * New_Z_cam, New_Z_cam, 1]).T
                New_point_world = np.dot(New_ex[used_index], New_point_cam)
                New_point_world_3d = New_point_world[:3] / New_point_world[3]
                #记录拉伸幅度
                lashen_move_x.append(New_point_world_3d[0] - point_world_3d[0])
                lashen_move_y.append(New_point_world_3d[1] - point_world_3d[1])


#-------------------------对于不同的模式，使用不同方法重建稠密点云

# 根据之前存储的偏移量，计算出平均偏移，然后应用到稠密点云
if mode == 'pingyi':
    pingyi_pianyi = np.array(pingyi_offset)
    mean_x = np.mean(pingyi_pianyi[:,0])
    mean_y = np.mean(pingyi_pianyi[:,1])
    mean_z = np.mean(pingyi_pianyi[:,2])
    means = np.array([mean_x, mean_y, mean_z])
    means = np.expand_dims(means, axis=0)

    print('pianyi:', means)

    #中心化的点 + 平移偏移量 + 去中心化
    points_new_all = points_dense_center + means + np.mean(points_dense, axis=0)

#缩放只需要尺度因子对整体缩放即可
elif mode == 'suoxiao' or mode == 'fangda':

    points_new_all = point_suofang(points_dense_center, suofang_scale)

    #中心化 + 去中心化
    points_new_all = points_new_all + np.mean(points_dense, axis=0)

#对于旋转，需要从GUI得到弧度，然后给定旋转中心，然后就可以计算了
elif mode == 'xuanzhuan':
    #弧度由gui得到，每次需要改变
    radian = -0.4957

    #旋转中心默认为mesh中心，后续再改
    center = np.array([0,0,0])

    rot = np.array([
    [np.cos(radian), -np.sin(radian), 0],
    [np.sin(radian),  np.cos(radian), 0],
    [0, 0, 1] ])

    #中心旋转 + 去中心化
    points_new_all = point_xuanzhuan(points_dense_center, center, rot) + np.mean(points_dense, axis=0)

#拉伸，根据光流垂线做出面
elif mode == 'lashen':
    #起始点和终点，方便计算垂线/面

    x1,y1,x2,y2 = 378,293,387,113
    move_max_x = np.array(lashen_move_x).max()
    move_max_y = np.array(lashen_move_y).max()

    #中心截面点云 + 去中心化
    points_new_all = point_lashen(points_dense_center, x1, y1, x2, y2, flow_numpy, move_max_x, move_max_y) + np.mean(points_dense, axis=0)

#--------------------------------------------------计算多视角光流(要用稠密点云)----------------------
#---------------计算点云法线，只使用正面计算点云


pcd_nor_old = o3d.geometry.PointCloud()

pcd_nor_old.points = o3d.utility.Vector3dVector(points_dense)

pcd_nor_old.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

normals_old = np.asarray(pcd_nor_old.normals)

centroid = np.mean(points_dense, axis=0) #点云中心
vectors_to_centroid = points_dense - centroid #每个点的方向

# 计算每个指向质心的向量与对应法线的点积
dot_products = np.sum(vectors_to_centroid * normals_old, axis=1)
# 如果点积是负的，说明法线指向了质心，应该翻转它 ---根据方向校正法线
flip_normals = dot_products < 0
normals_old[flip_normals] *= -1

o3d.io.write_point_cloud("pcd_with_normals.ply", pcd_nor_old)

nor_num = 0
#------------对于每个点，根据法线计算计算光流
#points_all和points_new_all，是原始稠密点云和运动稠密点云
for point_world_3d, New_point_world_3d in zip(points_dense, points_new_all):

    i = 0
    point_a = np.append(point_world_3d, 1)
    point_b = np.append(New_point_world_3d, 1)
    normals_point = normals_old[nor_num]  # 法线

    #对于每个视角，都进行该点的光流计算
    for index, new_id in enumerate(points_id):

        #首先计算相机坐标系的点
        point_cam_a = np.dot(np.linalg.inv(New_ex[index]), point_a)
        point_cam_b = np.dot(np.linalg.inv(New_ex[index]), point_b)

        #相机的平移、位置
        camera_position = New_ex[index][:3, 3]
        point_to_camera_vectors = camera_position - point_world_3d

        #法线点积
        dot_products = np.dot(normals_point, point_to_camera_vectors)

        #只对正面点云进行光流计算
        #if dot_products>=0:

        u_a = (point_cam_a[0] * fx / point_cam_a[2]) + cx
        v_a = (point_cam_a[1] * fy / point_cam_a[2]) + cy

        u_b = (point_cam_b[0] * fx / point_cam_b[2]) + cx
        v_b = (point_cam_b[1] * fy / point_cam_b[2]) + cy

        #在其他视角有效深度区域计算
        #只计算有效起始点的光流, 在域内且有深度
        if 0<= u_a <512 and 0<= v_a <512:
            flows_views[i][int(v_a),int(u_a), 0] = u_b - u_a
            flows_views[i][int(v_a),int(u_a), 1] = v_b - v_a

        i = i + 1
    #八个视角计算完，到下一个点云法线
    nor_num = nor_num + 1


###-----------得到新的mask
for flow, mask_id in zip(flows_views, points_id):
    mask = np.zeros((512,512),dtype=np.uint8)
    #所有flow区域都设为mask,判断条件是非0区域
    mask[np.any(flow!=0, axis=2)] = 255
    new_mask = expand_mask_with_flow(mask, flow, 1.0)
    inverted_mask = 255 - new_mask  # 翻转颜色
    downsampled_image = cv2.resize(inverted_mask, (64, 64))
    black_image = np.zeros_like(downsampled_image)+255

    # 遍历每个像素，如果它不是白色（即RGB值不全为255），则将其设为黑色
    for i in range(downsampled_image.shape[0]):
        for j in range(downsampled_image.shape[1]):
            if not np.all(downsampled_image[i, j] == 255):
                black_image[i, j] = [0, 0, 0]

    tensor_image = torch.from_numpy(black_image)
    transposed_tensor = tensor_image.permute(2, 0, 1)  # (3,64,64)
    new_channel = transposed_tensor[2, :, :].unsqueeze(0)  # 获取第三个通道的值并添加一个维度
    new_tensor = torch.cat([transposed_tensor, new_channel], dim=0)
    new_tensor = new_tensor.unsqueeze(0)

    bool_torch_image = new_tensor.bool()
    torch.save(bool_torch_image, 'mask' + str(mask_id) + '.pth')
#-----------------------------------------------------计算3D光流与2D多视角光流

flow_id = 0
for id in points_id:
    # 保存最终结果
    result_tensor = torch.from_numpy(flows_views[flow_id]).unsqueeze(0).permute(0, 3, 1, 2)  # 假设数据存储顺序需要修改
    # 保存张量到 .pth 文件
    # (1,2,512,512) tensor
    result_tensor = result_tensor.float()
    torch.save(result_tensor, 'flow' + str(id) + '.pth')
    flow_id = flow_id + 1


# 设置第一组点云的颜色为红色
red_color = [1, 0, 0]  # RGB颜色
pcd_old.points = o3d.utility.Vector3dVector(points_dense)
pcd_old.colors = o3d.utility.Vector3dVector(np.tile(red_color, (len(points_dense), 1)))

# 设置第二组点云的颜色为蓝色
blue_color = [0, 0, 1]  # RGB颜色
pcd_old.points.extend(points_new_all)
pcd_old.colors.extend(np.tile(blue_color, (len(points_new_all), 1)))

o3d.io.write_point_cloud("old_cloud.ply", pcd_old)
