"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
pytorch data loader for the 12-scenes dataset
"""
import os
import os.path as osp
import numpy as np
from torch.utils import data
from utils import load_image, load_depth, load_point
import sys
import pickle

sys.path.insert(0, '../')
from common.pose_utils import process_poses

class TwelveScenes(data.Dataset):
    def __init__(self, scene, data_path, train, \
    transform=None, depth_transform=None, dn_transform=None, target_transform=None, mode=0, d_suffix='depth', seed=7, real=False,skip_images=False, vo_lib='orbslam', draw_seq=None):
      """
      :param scene: scene name ['chess', 'pumpkin', ...]
      :param data_path: root 7scenes data directory.
      Usually '../data/deepslam_data/7Scenes'
      :param train: if True, return the training images. If False, returns the
      testing images
      :param transform: transform to apply to the color images
      :param depth_transform: depth_transform to apply to the color images
      :param target_transform: transform to apply to the poses
      :param mode: 0: just color image, 1: just depth image, 2: [c_img, d_img]
      :param d_suffix: only valid when using 7Scenes under mode 1,2 and 3 , choices=('depth', 'full_depth', 'full_d_cmap')
      :param real: If True, load poses from SLAM/integration of VO
      :param skip_images: If True, skip loading images and return None instead
      :param vo_lib: Library to use for VO (currently only 'dso')
      :param draw_seq: only for drawing sequence's trajectory, see scripts/plot_dataset.py or scripts/plot_train_test.py
      """
      self.mode = mode
      self.transform = transform
      self.depth_transform = depth_transform
      self.dn_transform = dn_transform
      self.target_transform = target_transform
      self.skip_images = skip_images
      self.d_suffix = d_suffix
      np.random.seed(seed)

      # directories
      base_dir = osp.join(osp.expanduser(data_path), scene)
      data_dir = osp.join('..', 'data', '12Scenes', scene)

      # decide which sequences to use
      if train:
        split_file = osp.join(base_dir, 'TrainSplit.txt')
      else:
        split_file = osp.join(base_dir, 'TestSplit.txt')
      with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

      # only for drawing sequence's trajectory
      if draw_seq is not None:
        seqs = [draw_seq]

      # read poses and collect image names
      self.c_imgs = []
      self.d_imgs = []
      self.gt_idx = np.empty((0,), dtype=np.int)
      ps = {}
      vo_stats = {}
      gt_offset = int(0)
      for seq in seqs:
        seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
        seq_data_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
        p_filenames = [n for n in sorted(os.listdir(osp.join(seq_dir, '.'))) if
                       n.endswith('pose.txt')]

        frame_idx = np.array(xrange(len(p_filenames)), dtype=np.int)
        pss = [np.loadtxt(osp.join(seq_dir, p_filename)).flatten()[:12] for p_filename in p_filenames]
        ps[seq] = np.asarray(pss)
        vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx))
        gt_offset += len(p_filenames)
        # c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
        #           for i in frame_idx]
        c_imgs = [osp.join(seq_dir, n) for n in sorted(os.listdir(osp.join(seq_dir, '.'))) if n.endswith('color.jpg')]          
        if self.d_suffix == 'cam_points' or self.d_suffix.find('scn_points') >= 0:
          d_fn = self.d_suffix + '.npy'
        else:
          d_fn = self.d_suffix + '.png'
        # d_imgs = [osp.join(seq_dir, 'frame-{:06d}.{:s}'.format(i, d_fn))
        #           for i in frame_idx]
        d_imgs = [osp.join(seq_dir, n) for n in sorted(os.listdir(osp.join(seq_dir, '.'))) if n.endswith(d_fn)]
        self.c_imgs.extend(c_imgs)
        self.d_imgs.extend(d_imgs)

      pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
      if train and not real:
        mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
        std_t = np.ones(3)
        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
      else:
        mean_t, std_t = np.loadtxt(pose_stats_filename)

      # convert pose to translation + log quaternion
      self.poses = np.empty((0, 6))
      for seq in seqs:
        pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
          align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
          align_s=vo_stats[seq]['s'])
        self.poses = np.vstack((self.poses, pss))

    def __getitem__(self, index):
      if self.skip_images:
        img = None
        pose = self.poses[index]
      else:
        if self.mode == 0:
          img = None
          while img is None:
            # img = {'c': load_image(self.c_imgs[index]), 'd': load_depth(self.d_imgs[index])}
            img = {'c': load_image(self.c_imgs[index])}
            pose = self.poses[index]
            index += 1
          index -= 1
        elif self.mode == 1:
          img = None
          while img is None:
            if self.d_suffix == 'cam_points' or self.d_suffix.find('scn_points') >= 0 : 
              img = {'d':  load_point(self.d_imgs[index])}
            else:
              img = {'d':  load_depth(self.d_imgs[index])}
            pose = self.poses[index]
            index += 1
          index -= 1
        else:
          c_img = None
          d_img = None
          while (c_img is None) or (d_img is None):
            c_img = load_image(self.c_imgs[index])
            # load_image use pil_loader which convert d_image into RGB (3 channels, 8 bits per channel), while d_img is 16 bits png image which should be convert into I (32 bits, 1 channel)
            if self.d_suffix == 'cam_points' or self.d_suffix.find('scn_points') >= 0 :
              d_img = load_point(self.d_imgs[index]) 
            else:
              d_img = load_depth(self.d_imgs[index])
            # d_img = load_image(self.d_imgs[index])
            pose = self.poses[index]
            index += 1
          img = {'c': c_img, 'd': d_img}
          index -= 1


      if self.target_transform is not None:
        pose = self.target_transform(pose)

      if self.skip_images:
        return img, pose

      if self.mode >= 2 and self.transform is not None and self.depth_transform is not None and self.dn_transform is not None:
        img = {'c': self.transform(img['c']), 'd': self.depth_transform(img['d']), 'dn': self.dn_transform(img['d'])}
      elif self.mode == 1 and self.dn_transform is not None:
        img = {'dn': self.dn_transform(img['d'])}
      elif self.mode == 0 and self.transform is not None:
        img = {'c': self.transform(img['c'])}
        # img = {'c': self.transform(img['c']), 'd': self.depth_transform(img['d'])}
      else:
        raise Exception('Missing transform for mode {:d}'.format(self.mode))
      return img, pose

    def __len__(self):
      return self.poses.shape[0]

def main():
  """
  visualizes the dataset
  """
  from common.vis_utils import show_batch, show_stereo_batch
  from torchvision.utils import make_grid
  import torchvision.transforms as transforms
  seq = 'chess'
  mode = 0
  num_workers = 6
  transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #   std=[0.229, 0.224, 0.225])
  ])
  dset = SevenScenes(seq, '../data/deepslam_data/7Scenes', True, transform,
    depth_transform=transform, mode=mode)
  print 'Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq,
    len(dset))

  data_loader = data.DataLoader(dset, batch_size=10, shuffle=True,
    num_workers=num_workers)

  batch_count = 0
  N = 2
  for (imgs, poses) in data_loader:
    print 'Minibatch {:d}'.format(batch_count)
    if mode == 0:
      show_batch(make_grid(imgs['c'], nrow=1, padding=25))
    elif mode == 1:
      show_batch(make_grid(imgs['d'], nrow=1, padding=25))
    elif mode == 2:
      lb = make_grid(imgs['c'], nrow=1, padding=25)
      rb = make_grid(imgs['d'], nrow=1, padding=25)
      show_stereo_batch(lb, rb)

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
  main()
