import scipy
import skimage
from skimage import io
from skimage import transform
from scipy import ndimage
import numpy as np
from PIL import Image
import os
import os.path as osp
import matplotlib.pyplot as plt
import argparse

# generate scene 3d point coordinates. size: [h, w]
def depth_to_3dpoint(pose, depth, h, w, depth_scale, I=None):
    if (I is None):
        I = np.array([[585, 0, 320],
                                    [0, 585, 240],
                                    [0, 0, 1]])
    downscale_u = 640.0 / w
    downscale_v = 480.0 / h
    I = np.concatenate((I[0:1, :]/downscale_u, I[1:2, :]/downscale_v, I[2:, :]), axis=0)
    I_inv = np.linalg.inv(I)

    u_range, v_range = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
    grid_u, grid_v = np.meshgrid(u_range, v_range)
    grid_ones = np.ones((h, w))
    uv_coords = np.stack((grid_u, grid_v, grid_ones), axis=0) # [3, H, W]
    f_coords = np.matmul(I_inv, uv_coords.reshape(3, -1))
    D = depth.reshape(1, h*w) / depth_scale
    D = np.concatenate((D, D, D), axis=0)
    cam_coords = f_coords*D # [3, h*w]

    rot = pose[: 3, :3]
    trans = pose[:3, 3:]
    scn_coords = np.matmul(rot, cam_coords) + trans
    return scn_coords.reshape(3, h, w)

def main(root_dir, val, h=256, w=341, depth_scale=1000):
    if val:
        split_file = osp.join(root_dir, 'TestSplit.txt')
    else:
        split_file = osp.join(root_dir, 'TrainSplit.txt')

    with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

    fulld_imgs = []
    pose_txt = []
    for seq in seqs:
        seq_dir = osp.join(root_dir, 'seq-{:02d}'.format(seq))
        p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('pose') >= 0]
        frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
        pose_fn = [osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(i))
                  for i in frame_idx]
        fulld = [osp.join(seq_dir, 'frame-{:06d}.full_depth.png'.format(i))
                  for i in frame_idx]
        fulld_imgs.extend(fulld)
        pose_txt.extend(pose_fn)

    for i, (pose_fn, fulld_img) in enumerate(zip(pose_txt, fulld_imgs)):
        if i % 200 == 0:
            print( 'Image {:d} / {:d}'.format(i, len(fulld_imgs))) 
        fulld = skimage.io.imread(fulld_img, as_gray=True)
        fulld_resized = skimage.transform.resize(fulld,(h, w), preserve_range=True)
        pose = np.loadtxt(pose_fn)
        # d[d>3600] = 0
        coords = depth_to_3dpoint(pose, fulld_resized, h, w, depth_scale)
        output_path = fulld_img.replace('full_depth.png', 'scn_points.npy')
        np.save(output_path, coords)

def test(h=256, w=341, depth_scale=1000.0):
    root_dir = './chess/seq-01/'
    p0 = np.loadtxt(root_dir + 'frame-000000.pose.txt')
    p10  = np.loadtxt(root_dir + 'frame-000030.pose.txt')
    R = np.matmul(np.linalg.inv(p0), p10)
    print(R)
    rot = R[: 3, :3]
    trans = R[:3, 3:]
    img0 = skimage.io.imread(root_dir + 'frame-000000.color.png')
    img0 = skimage.transform.resize(img0,(h, w), preserve_range=True)
    img10 = skimage.io.imread(root_dir + 'frame-000030.color.png')
    img10 = skimage.transform.resize(img10,(h, w), preserve_range=True)
    depth10 = skimage.io.imread(root_dir + 'frame-000030.depth.png', as_gray=True)
    depth10 = skimage.transform.resize(depth10,(h, w), preserve_range=True)
    #####  depth_to_3dpoint BEGIN
    # cam10, I = depth_to_3dpoint(depth10, h, w, depth_scale)
    depth = depth10
    ###
    I = np.array([[585, 0, 320],
                                    [0, 585, 240],
                                    [0, 0, 1]])
    downscale_u = 640.0 / w
    downscale_v = 480.0 / h
    I = np.concatenate((I[0:1, :]/downscale_u, I[1:2, :]/downscale_v, I[2:, :]), axis=0)
    I_inv = np.linalg.inv(I)
    
    u_range, v_range = np.linspace(0, w-1, w), np.linspace(0, h-1, h)
    grid_u, grid_v = np.meshgrid(u_range, v_range)
    grid_ones = np.ones((h, w))
    uv_coords = np.stack((grid_u, grid_v, grid_ones), axis=0) # [3, H, W]
    print(uv_coords)
    f_coords = np.matmul(I_inv, uv_coords.reshape(3, -1))
    D = depth.reshape(1, h*w) / depth_scale
    D = np.concatenate((D, D, D), axis=0)
    print(np.max(D))
    cam_coords = f_coords*D # [3, h*w]
    print('cam_coords', cam_coords)
    print(np.max(cam_coords), np.min(cam_coords), np.mean(cam_coords))
    ###
    cam10 = cam_coords
    #####  depth_to_3dpoint END
    proj10 = np.matmul(rot, cam10) + trans
    print(proj10.shape)
    warp10 = np.matmul(I, proj10)
    print(warp10.shape)
    src_x, src_y, src_z = warp10[0], warp10[1], warp10[2] # [h*w]
    src_z = np.clip(src_z, a_min=1e-10, a_max=None)
    print(src_x.shape, src_z.shape)
    # src_u = 2 * (src_x / src_z) / (w - 1) - 1
    # src_v = 2 * (src_y / src_z) / (h - 1) - 1
    src_u = src_x / src_z
    src_v = src_y / src_z
    warp_coords = np.stack([src_v, src_u], axis=0).reshape(2, h, w)  # [2, h, w]
    print('warp coords', warp_coords)
    # warped10 = skimage.transform.warp(img0, warp_coords)
    r = ndimage.map_coordinates(img0[..., 0], warp_coords)
    g = ndimage.map_coordinates(img0[..., 1], warp_coords)
    b = ndimage.map_coordinates(img0[..., 2], warp_coords)
    warped10 = np.uint8(np.stack([r, g, b], axis=2))
    img10 = np.uint8(img10)
    img0 = np.uint8(img0)
    print(warped10)
    plt.imshow(np.concatenate((img0, img10, warped10), axis=1), interpolation='nearest')
    # plt.imshow(warped10, interpolation='nearest')
    # plt.imshow(img10, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script is Python port of depth filling code from NYU toolbox  
    ''')
    # parser.add_argument('--root_dir', default='./chess', help='scene\'s root directory')
    parser.add_argument('root_dir', help='scene\'s root directory')
    parser.add_argument('--val', action='store_true', help='scene\'s root directory')
    args = parser.parse_args()
    
    main(args.root_dir, args.val)
    # test()

