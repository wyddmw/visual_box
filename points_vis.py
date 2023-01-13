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
#        depth = torch.clamp(depth, 0, 80.0)        # set distance threshold within 80m
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
    import imageio
    imageio.imwrite('projected_depth.png', projected_depth)

def kitti_sample_world_coords(right_image, left_image, left_disp, scale=2):
    # project right view to left view given left view direction
    # scale test
    mask = (left_disp > 0) & (left_disp < 192)
    # mask = F.avg_pool2d(mask.float(), (4, 4), stride=2) > 0.25
    # right_image = F.avg_pool2d(right_image, (2, 2), stride=2)
    # left_image = F.avg_pool2d(left_image, (2, 2), stride=2)
    # left_disp = F.avg_pool2d(left_disp, (2, 2), stride=2) / scale
    right_image = F.interpolate(right_image, scale_factor=0.5, mode='bilinear')
    left_image = F.interpolate(left_image, scale_factor=0.5, mode='bilinear')
    left_disp = F.interpolate(left_disp[None], scale_factor=0.5, mode='bilinear') / scale
    mask = F.interpolate(mask[None].float(), scale_factor=0.5, mode='bilinear') > 0.5
    # left_disp = left_disp * mask + 3

    focal_length = 721.277 / scale
    N, C, H, W = right_image.shape
    left_depth = (0.54 * focal_length) / (left_disp + 1e-6)
    left_depth = left_depth.view(-1, 1)
    # convert camera coords to world coords
    camera_coords = generate_meshgrid(left_disp).view(-1, 2)
    
    # compute right world coods
    world_coords_xy = camera_coords / focal_length * left_depth
    world_coords_xy[:, 0] = world_coords_xy[:, 0] - 0.54
    world_coords = torch.cat((world_coords_xy, left_depth), dim=-1)
    right_sample_coords = world_coords[:, :-1] / world_coords[:, -1:] * focal_length

    right_sample_coords = right_sample_coords.view(1, H, W, -1).permute(0, 3, 1, 2)
    right_image_sampled = normalize_coords(right_sample_coords)
    warped_img = F.grid_sample(right_image, right_image_sampled, mode='nearest', padding_mode='zeros')
    warped_img = warped_img * mask
    
    # warping by disparity
    warped_left, valid_mask = disp_warp(right_image, left_disp)
    warped_left = warped_left * mask  
    TensorToPILImage(warped_left[0], saving_path='disp_warped_right.png')

    TensorToPILImage(warped_img[0], saving_path='warped_right.png')
    TensorToPILImage(left_image[0], saving_path='scale_left.png')

def kitti_sample_pipeline(right_image, left_image, left_disp, scale=1):
    mask = (left_disp > 0) & (left_disp < 192)
    H, W = right_image.shape[-2:]
    baseline = 0.54
    focal_length = 721.277
    if scale > 1:
        mask = F.avg_pool2d(mask.float(), (2, 2), stride=2) > 0
        right_image = F.avg_pool2d(right_image, (2, 2), stride=2)
        left_image = F.avg_pool2d(left_image, (2, 2), stride=2)
        left_disp = F.avg_pool2d(left_disp, (2, 2), stride=2) / scale
        focal_length = focal_length / scale
    
    K = torch.Tensor(np.array([
        [721.277, 0., 0.5 * H],
        [0., 721.277, 0.5 * W],
        [0., 0., 1.]
    ]))

    Rt = torch.Tensor(np.array([
        [1, 0, 0, -baseline],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]))

    direction = get_ray_directions(left_disp, focal_length)
    rays_d, rays_o = get_rays(direction, Rt)
    
    depth = focal_length * baseline / (left_disp + 1e-6).view(-1, 1)
    world_coord = rays_o + depth * rays_d
    
    # world to pixel coord
    world_coord[:, 0] = world_coord[:, 0] / world_coord[:, -1] * focal_length + 0.5 * H
    world_coord[:, 1] = world_coord[:, 1] / world_coord[:, -1] * focal_length + 0.5 * W
    camera_coord = world_coord[:, :-1]
    camera_coord = camera_coord.view(1, H, W, -1).permute(0, 3, 1, 2)
    right_image_sampled = normalize_coords(camera_coord)
    
    warped_img = F.grid_sample(right_image, right_image_sampled, mode='bilinear', padding_mode='zeros')
    warped_img = warped_img * mask
    TensorToPILImage(warped_img[0], saving_path='warped_right.png')

def get_ray_directions(left_disp, focal, center=None):
    H, W = left_disp.shape[-2:]

    camera_coords = generate_meshgrid(left_disp).view(-1, 2)
    center = [H/2, W/2]
    # convert pixel coord to camera coord

    camera_coords[:, 0] = camera_coords[:, 0] - center[0]
    camera_coords[:, 1] = camera_coords[:, 1] - center[1]
    direction = camera_coords / focal
    
    direction  = torch.cat((direction, torch.ones(camera_coords.shape[0], 1)), dim=-1)
    
    return direction

def get_rays(direction, c2w):
    rays_d = direction @ c2w[:3, :3].T
    rays_o = c2w[:3, -1].expand(rays_d.shape)       # H, W, 3
    # rays_d = rays_d.view(-1, 3)
    # rays_o = rays_o.view(-1, 3)
    return rays_d, rays_o
    

def get_patches(rays_all, patch_size, num_patches, disp):
    '''
    Get patches from images.
    Args:
        Rays_all: [H, W, 3, 3] -> [H, W, r_o+r_d+rgb, 3]
        Patch_size: int
        Num_patches: num_rays / patch_size ** 2
        Disp: [H, W]
    Return:
        selected_rays: [num_patches, patch_size**2, 3, 3]
        selected_disp: [num_patches, patch_size**2, 1]
    '''
    # prevent sampling from invalid points out of depth threshold
    H, W, _, _ = rays_all.shape
    if len(disp.shape) > 2:
        disp = disp.reshape(H, W)
    disp_mask = (disp > 0) & (disp < 192)       # [H, W]
    disp_mask = disp_mask.reshape(-1)
    
    meshgrid = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1).reshape(-1, 2)
    
    valid_grid = meshgrid[disp_mask]
    
    grid_neighbors = valid_grid[:, None, :] + np.stack(
      np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
      axis=-1).reshape(1, -1, 2)
    # applying boundary threshold
    boundary_mask = (grid_neighbors[..., 0].max(axis=-1) < H) & (grid_neighbors[..., 1].max(axis=-1) < W)
    # clip boundaries
    grid_neighbors[..., 0] = np.clip(grid_neighbors[..., 0], 0, H-1)
    grid_neighbors[..., 1] = np.clip(grid_neighbors[..., 1], 0, W-1)
    disp_mask = disp_mask.reshape(H, W)
    center_masks = disp_mask[grid_neighbors[..., 0], grid_neighbors[..., 1]].sum(axis=-1) == (patch_size ** 2)    # [num_valid_centers, patch_size*patch_size]
    center_masks = center_masks * boundary_mask
    # depth value for selected patches should be valid
    valid_grid = valid_grid[center_masks]           # [num_valid_centers, 2]
    
    selected_index = np.random.randint(0, valid_grid.shape[0], size=(num_patches))
    selected_grid = valid_grid[selected_index, None, :]
    
    patch_idx = selected_grid + np.stack(
      np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
      axis=-1).reshape(1, -1, 2)
    selected_images = rays_all[patch_idx[..., 0], patch_idx[..., 1]]
    selected_disp = disp[patch_idx[..., 0], patch_idx[..., 1]]
    return selected_images, selected_disp


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
        
        # generate projected depth
        # kitti_project_depth(disp_tensor[:, None])

        # pixel sample given coords
        # kitti_sample_world_coords(right_tensor[None], left_tensor[None], disp_tensor)

        # pixel sample following a standard rendering pipeline
        kitti_sample_pipeline(right_tensor[None], left_tensor[None], disp_tensor)
        # prepare_kitti_dataset(left_tensor,right_tensor, disp_tensor, )
        # generate patches
        # left_tensor = left_tensor.permute(1, 2, 0).unsqueeze(dim=-2)
        # selected_patches, selected_disp = get_patches(left_tensor, 16, 4, disp=disp_tensor)
        
        # batches = selected_patches.shape[0]
        # selected_patches = selected_patches.reshape(4, 16, 16, -1, 3).permute(0, 3, 4, 1, 2)
        
        # for i in range(batches):
        #     TensorToPILImage(selected_patches[i][0], img_show=True)
