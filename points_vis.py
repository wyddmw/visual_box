import open3d
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import argparse


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
        depth_mask = disp_gt > 0
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

def TensorToPILImage(img_tensor, saving_path=None, img_show=False):
    image = transforms.ToPILIMage().convert('RGB')
    if saving_path is not None:
        image.save(saving_path)
    if img_show:
        image.show()
    return image

def TensorToNumpy(img_tensor):
    image = img_tensor.data.cpu().numpy() * 255.
    image = image.astype('np.uint8')
    assert len(image.shape) == 3
    image = np.transpose(image, (1, 2, 0))  # H, W C
    if saving_path is not None:
        cv2.imwrite(image, saving_path)
    if img_show:
        cv2.imshow('image', image)
        cv2.waitKey(0)
    return image

def generate_meshgrid(img):
    H, W = img.shape[-2:]
    x_range = torch.arange(0, W).view(1, 1, W).expand(1, H, W)
    y_range = torch.arange(0, H).view(1, H, 1).expand(1, H, W)
    grid = torch.cat((x_range, y_range), dim=0).permute(1, 2, 0)    # [H, W, 2]
    return grid
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_img', type=str, default='./left.png')
    parser.add_argument('--right_img', type=str, default='./right.png')
    parser.add_argument('--disp', type=str, default=None)
    parser.add_argument('--flow', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
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
    if args.disp is not None:
        disp = np.array(Image.open(args.disp))
        disp = disp.astype(np.float32) / 256.
        disp_tensor = transform(disp)
    prepare_kitti_dataset(left_tensor, right_tensor, disp_gt=disp_tensor, vis_points=True)
