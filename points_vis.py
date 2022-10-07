from email import iterators
import open3d
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import argparse

from utils import *


def draw_scenes(points, draw_origin=False, point_colors=False):
    if isinstance(points, torch.Tensor):
        points = points.data.cpu().numpy()
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.ones(3) * 255
    
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)
    
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)
    if point_colors:
        pts.colors = open3d.utility.Vector3dVector(points[:, 3:6])
    
    vis.run()
    vis.destroy_window()

def prepare_kitti_dataset(left_img, right_img, disp_gt=None, flow_gt=None, vis_points=False):
    '''
    :param left_img input tensor [N, 3, H, W]
    :param right_img input tensor [N, 3, H, W]
    :disp_gt disparity [N, H, W]
    '''
    focal_length = 721
    baseline = 0.54
    H, W = left_img.shape[-2:]
    if disp_gt is not None:
        # convert disparity to depth
        depth = baseline * focal_length / (disp_gt + 1e-4)
        depth = depth.view(-1, 1)
#        depth_mask = disp_gt > 0
        depth_mask = depth < 50.
        depth_mask = depth_mask.reshape(-1)
#        depth = torch.clamp(depth, 0, 80.0)     # set distance threshold within 80m
    meshgrid = generate_meshgrid(disp).view(-1, 2)  # [H*W, 2]
    pts = torch.ones((H*W, 6))
    pts[:, 0] = (meshgrid[:, 0] - W / 2) / focal_length 
    pts[:, 1] = (meshgrid[:, 1] - H / 2) / focal_length
    pts[:, :3] = pts[:, :3] * depth 
    pts[:, 3:] = left_img.view(-1, 3)
    if depth_mask is not None:
        pts[~depth_mask] = 0
    draw_scenes(pts, point_colors=False, draw_origin=True)

def kitti_project_depth(src_depth):
    N, C, H, W = src_depth.shape
    src_depth = src_depth.reshape(-1, 1)
    meshgrid = generate_meshgrid(disp).view(-1, 2)  # [H*W, 2]
    
    src_depth = (0.54 * 721.277) / (src_depth + 1e-5)
    pts = meshgrid / 721.277 * src_depth
    pts = torch.cat((pts, src_depth), dim=-1)

    # convert to target view
    pts[:, 0] = pts[:, 0] - 0.54
    right_coords = pts[:, :-1] / pts[:, -1:] * 721.277
    right_coords = right_coords.long()
    valid_mask = (pts[:, -1] < 80.0) & (right_coords[:, 0] < W) & (right_coords[:, 1] < H)
    
    right_coords = right_coords.reshape(-1, 2)
    
    projected_depth = torch.zeros_like(src_depth).view(N, H, W)        # [N, C, H, W]
    
    projected_depth[:, right_coords[valid_mask, 1], right_coords[valid_mask, 0]] = 0.54 * 721.277 / pts[valid_mask, -1].unsqueeze(dim=0)
    
    projected_depth = projected_depth.view(H, W, 1).cpu().numpy().astype(np.uint8)
    import cv2
    erode_kernel = np.ones((3, 3), np.uint8)
    projected_depth = cv2.dilate(projected_depth, erode_kernel, iterations=1)
    # import imageio
    # imageio.imwrite('projected_depth.png', projected_depth)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_img', type=str, default=None)
    parser.add_argument('--right_img', type=str, default=None)
    parser.add_argument('--disp', type=str, default=None)
    parser.add_argument('--disp_right', type=str, default=None)
    parser.add_argument('--flow', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.left_img is not None:
        left_img = Image.open(args.left_img).convert('RGB')
        right_img = Image.open(args.right_img).convert('RGB')
    
        train_transform_list = [#transforms.RandomCrop(args.img_height, args.img_width),
                                transforms.ToTensor(),
    #                            #transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                                ]
    #
        transform = transforms.Compose(train_transform_list)
        
        left_tensor = transform(left_img)
        right_tensor = transform(right_img)

    if args.flow is not None:
        flow = readFlow(args.flow).astype(np.float32)
        flow_vis_name = 'flow.png'
        
        save_vis_flow_tofile(flow, flow_vis_name, vis=False)

        # image warping 
        flow = torch.Tensor(flow)
        if len(left_tensor.shape) == 3:
            left_tensor = left_tensor.unsqueeze(dim=0)
            right_tensor = right_tensor.unsqueeze(dim=0)
        flow = flow.unsqueeze(dim=0).permute(0, 3, 1, 2)
        warped_img = flow_warp(right_tensor, flow)
        error_map = torch.abs((warped_img - left_tensor)).squeeze(dim=0)
        warped_img = warped_img.squeeze(dim=0)
        
        TensorToPILImage(error_map, img_show=True)
        TensorToPILImage(warped_img, img_show=True)

    if args.disp is not None:
        disp = np.array(Image.open(args.disp))
        disp = disp.astype(np.float32) / 256.
        disp_tensor = transform(disp)
        kitti_project_depth(disp_tensor[:, None])
        