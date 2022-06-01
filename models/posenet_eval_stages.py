"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
implementation of PoseNet and MapNet networks 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ['TORCH_MODEL_ZOO'] = os.path.join('..', 'data', 'models')

import sys
sys.path.insert(0, '../')

#def trace_hook(m, g_in, g_out):
#  for idx,g in enumerate(g_in):
#    g = g.cpu().data.numpy()
#    if np.isnan(g).any():
#      set_trace()
#  return None

def filter_hook(m, g_in, g_out):
  g_filtered = []
  for g in g_in:
    g = g.clone()
    g[g != g] = 0
    g_filtered.append(g)
  return tuple(g_filtered)

def fibonacci_sphere(samples=64):

  points = []
  phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

  for i in xrange(samples):
    z = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - z * z)  # radius at y

    theta = phi * i  # golden angle increment

    x = np.cos(theta) * radius
    y = np.sin(theta) * radius

    points.append((x, y, z))

  return np.array(points)

def main():
  fig = plt.figure()

  ax = fig.add_subplot(111, projection='3d')

  plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
  
  # start of Sphere base poses
  dir_samples = 256
  unit_points = fibonacci_sphere(dir_samples)
  u = np.linspace(0, 2 * np.pi, 100)
  v = np.linspace(0, np.pi, 100)
  x = np.outer(np.cos(u), np.sin(v))
  y = np.outer(np.sin(u), np.sin(v))
  z = np.outer(np.ones(np.size(u)), np.cos(v))
  
  mag_samples = 8
  t_range = np.linspace(0.001, 0.1, mag_samples)
  q_range = np.linspace(0.001, 0.5 * np.pi, mag_samples)
  points = np.outer(q_range, unit_points)
  print points.shape
  points = points.reshape(-1, 3)
  print points.shape
  # end of Sphere base poses
  '''
  # start of Minmal base poses
  points = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
  # end of Minmal base poses
  '''
  # ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='c', alpha=0.6)
  # ax.scatter(unit_points[:, 0], unit_points[:, 1], unit_points[:, 2],  label='base orientation',  s=20, color='k')
  ax.scatter(points[:, 0], points[:, 1], points[:, 2],  label='base orientation',  s=20, color='k')
  # ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='r')
  ax.view_init(azim=45, elev=30)

  ax.set_title('Fibonacci Sphere')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.legend()
  # ax.set_aspect("equal")
  # plt.tight_layout()
  plt.show(block=True)


class PoseNet_Alpha2(nn.Module):
  def __init__(self, feature_extractor, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0, isCube=False):
    super(PoseNet_Alpha2, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    self.two_stream_mode = two_stream_mode
    fe_out_planes = feature_extractor.fc.in_features
    if isCube:
    #******CUBE bases*********
      t_range = 0.01
      q_range = 0.01
      # base translations
      tx_range, ty_range, tz_range = torch.linspace(-1*t_range, t_range, 16), torch.linspace(-1*t_range, t_range, 8), torch.linspace(-1*t_range, t_range, 16)
      tx, ty, tz = torch.meshgrid([tx_range, ty_range, tz_range]) 
      self.base_t = torch.stack((tx, ty, tz), dim=-1).reshape(-1, 3).cuda()
      #base angles
      qx_range, qy_range, qz_range = torch.linspace(-1*q_range, q_range, 16), torch.linspace(-1*q_range, q_range, 8), torch.linspace(-1*q_range, q_range, 16)
      qx, qy, qz = torch.meshgrid([qx_range, qy_range, qz_range])
      self.base_q = torch.stack((qx, qy, qz), dim=-1).reshape(-1, 3).cuda()
    else:
    #******SPHERE bases*********
      # unit bases
      dir_samples = 64
      mag_samples = 32
      unit_points = fibonacci_sphere(dir_samples)
      # base translations
      t_range = np.linspace(0.01, 1, mag_samples)
      base_t = np.outer(t_range, unit_points).reshape(-1, 3)
      self.base_t = torch.from_numpy(base_t).float().cuda()
      #base angles
      q_range = np.linspace(0.01, 0.5 * np.pi, mag_samples)
      base_q = np.outer(q_range, unit_points).reshape(-1, 3)
      self.base_q = torch.from_numpy(base_q).float().cuda()

    if self.two_stream_mode == 0:
      # RGB feature extractor
      # replace the last FC layer in feature extractor
      self.rgb_feature_extractor = feature_extractor
      self.rgb_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.rgb_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.rgb_fc_xyz  = nn.Linear(feat_dim, feat_dim)
      self.rgb_fc_wpqr = nn.Linear(feat_dim, feat_dim)
    elif self.two_stream_mode == 1:
      # Depth feature extractor
      # replace the last FC layer in feature extractor
      self.depth_feature_extractor = feature_extractor
      self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.depth_fc_xyz  = nn.Linear(feat_dim, feat_dim)
      self.depth_fc_wpqr = nn.Linear(feat_dim, feat_dim)
    else:
      print 'two_stream_mode shoud be 0 or 1 at the first stage'
      raise NotImplementedError

    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      if self.two_stream_mode == 0:
        init_modules = [self.rgb_feature_extractor.fc, self.rgb_fc_xyz, self.rgb_fc_wpqr]
      else:
        init_modules = [self.depth_feature_extractor.fc, self.depth_fc_xyz, self.depth_fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, data):
    if self.two_stream_mode == 0:
      # RGB feature extractor
      x = self.rgb_feature_extractor(data)
      x = F.relu(x)
      if self.droprate > 0:
        x = self.dropout(x)
      # a_xyz  = self.rgb_fc_xyz(x)
      # a_log_q = self.rgb_fc_wpqr(x)
      a_xyz  = F.relu(self.rgb_fc_xyz(x))
      a_log_q = F.relu(self.rgb_fc_wpqr(x))
    else:
      # Depth feature extractor
      x = self.depth_feature_extractor(data)
      x = F.relu(x)
      if self.droprate > 0:
        x = self.dropout(x)
      # a_xyz  = self.depth_fc_xyz(x)
      # a_log_q = self.depth_fc_wpqr(x)
      a_xyz  = F.relu(self.depth_fc_xyz(x))
      a_log_q = F.relu(self.depth_fc_wpqr(x))
    output = {'at':a_xyz, 'bt':self.base_t, 'aq':a_log_q, 'bq':self.base_q}
    return output

class PoseNet_Alpha_Min(nn.Module):
  def __init__(self, feature_extractor, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0):
    super(PoseNet_Alpha_Min, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    self.two_stream_mode = two_stream_mode
    fe_out_planes = feature_extractor.fc.in_features
    # unit bases
    base = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
    self.base_t = torch.from_numpy(base).float().cuda()
    #base angles
    self.base_q = torch.from_numpy(base).float().cuda()

    if self.two_stream_mode == 0:
      # RGB feature extractor
      # replace the last FC layer in feature extractor
      self.rgb_feature_extractor = feature_extractor
      self.rgb_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.rgb_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.rgb_fc_xyz  = nn.Linear(feat_dim, 6)
      self.rgb_fc_wpqr = nn.Linear(feat_dim, 6)
    elif self.two_stream_mode == 1:
      # Depth feature extractor
      # replace the last FC layer in feature extractor
      self.depth_feature_extractor = feature_extractor
      self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.depth_fc_xyz  = nn.Linear(feat_dim, 6)
      self.depth_fc_wpqr = nn.Linear(feat_dim, 6)
    else:
      print 'two_stream_mode shoud be 0 or 1 at the first stage'
      raise NotImplementedError

    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      if self.two_stream_mode == 0:
        init_modules = [self.rgb_feature_extractor.fc, self.rgb_fc_xyz, self.rgb_fc_wpqr]
      else:
        init_modules = [self.depth_feature_extractor.fc, self.depth_fc_xyz, self.depth_fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, data):
    if self.two_stream_mode == 0:
      # RGB feature extractor
      x = self.rgb_feature_extractor(data)
      x = F.relu(x)
      if self.droprate > 0:
        x = self.dropout(x)
      # a_xyz  = self.rgb_fc_xyz(x)
      # a_log_q = self.rgb_fc_wpqr(x)
      a_xyz  = F.relu(self.rgb_fc_xyz(x))
      a_log_q = F.relu(self.rgb_fc_wpqr(x))
    else:
      # Depth feature extractor
      x = self.depth_feature_extractor(data)
      x = F.relu(x)
      if self.droprate > 0:
        x = self.dropout(x)
      # a_xyz  = self.depth_fc_xyz(x)
      # a_log_q = self.depth_fc_wpqr(x)
      a_xyz  = F.relu(self.depth_fc_xyz(x))
      a_log_q = F.relu(self.depth_fc_wpqr(x))
    output = {'at':a_xyz, 'bt':self.base_t, 'aq':a_log_q, 'bq':self.base_q}
    return output


class PoseNet_Alpha(nn.Module):
  def __init__(self, feature_extractor, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0):
    super(PoseNet_Alpha, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    self.two_stream_mode = two_stream_mode
    fe_out_planes = feature_extractor.fc.in_features

    if self.two_stream_mode == 0:
      # RGB feature extractor
      # replace the last FC layer in feature extractor
      self.rgb_feature_extractor = feature_extractor
      self.rgb_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.rgb_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.rgb_fc_xyz  = nn.Linear(feat_dim, 3)
      self.rgb_fc_wpqr = nn.Linear(feat_dim, 3)
    elif self.two_stream_mode == 1:
      # Depth feature extractor
      # replace the last FC layer in feature extractor
      self.depth_feature_extractor = feature_extractor
      self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.depth_fc_xyz  = nn.Linear(feat_dim, 3)
      self.depth_fc_wpqr = nn.Linear(feat_dim, 3)
    else:
      print 'two_stream_mode shoud be 0 or 1 at the first stage'
      raise NotImplementedError

    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      if self.two_stream_mode == 0:
        init_modules = [self.rgb_feature_extractor.fc, self.rgb_fc_xyz, self.rgb_fc_wpqr]
      else:
        init_modules = [self.depth_feature_extractor.fc, self.depth_fc_xyz, self.depth_fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, data):
    if self.two_stream_mode == 0:
      # RGB feature extractor
      x = self.rgb_feature_extractor(data)
      x = F.relu(x)
      if self.droprate > 0:
        x = self.dropout(x)
    else:
      # Depth feature extractor
      x = self.depth_feature_extractor(data)
      x = F.relu(x)
      if self.droprate > 0:
        x = self.dropout(x)
    return x


if __name__ == '__main__':
    main()