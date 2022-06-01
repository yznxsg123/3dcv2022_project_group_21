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
  '''
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
  points = np.outer(t_range, unit_points)
  print points.shape
  points = points.reshape(-1, 3)
  print points.shape
  # end of Sphere base poses
  '''
  # start of Min base poses
  points = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
  # end of Min base poses
  '''
  # start of Cube base poses
  t_range = 0.01
  tx_range, ty_range, tz_range = torch.linspace(-1*t_range, t_range, 16), torch.linspace(-1*t_range, t_range, 8), torch.linspace(-1*t_range, t_range, 16)
  tx, ty, tz = torch.meshgrid([tx_range, ty_range, tz_range]) 
  tx, ty, tz = tx.numpy(), ty.numpy(), tz.numpy()
  points = np.array((2048, 3))
  points  = np.concatenate((tx.reshape(-1,1), ty.reshape(-1,1), tz.reshape(-1,1)), axis=1)
  print points.shape
  # end of Cube base poses
  '''
  # ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='c', alpha=0.6)
  # ax.scatter(unit_points[:, 0], unit_points[:, 1], unit_points[:, 2],  label='base orientation',  s=20, color='k')
  ax.scatter(points[:, 0], points[:, 1], points[:, 2],  label='base orientation',  s=20, c=points[:, 0], cmap='tab20b', depthshade=False)#)
  # ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='r')
  ax.view_init(azim=45, elev=30)
  plt.savefig('../min.png')
  # ax.set_title('Fibonacci Sphere')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  # ax.legend()
  ax.set_aspect("equal")
  # plt.tight_layout()
  plt.show(block=True)

class PoseNet_Basepose(nn.Module):
  def __init__(self, feature_extractor, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0, base='mixed', bias=None):
    super(PoseNet_Basepose, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    self.two_stream_mode = two_stream_mode
    fe_out_planes = feature_extractor.fc.in_features
    if bias is not None:
      self.bias = torch.from_numpy(bias).cuda().float()
    else:
      self.bias = None
    if base == 'cube':
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
    elif base == 'sphere':
    #******SPHERE bases*********
      # unit bases
      dir_samples = 256
      mag_samples = 8
      unit_points = fibonacci_sphere(dir_samples)
      # base translations
      t_range = np.linspace(0.001, 0.1, mag_samples)
      # t_range = np.linspace(0.01, 1, mag_samples)
      base_t = np.outer(t_range, unit_points).reshape(-1, 3)
      self.base_t = torch.from_numpy(base_t).float().cuda()
      #base angles
      q_range = np.linspace(0.001, 0.1 , mag_samples)
      # q_range = np.linspace(0.01, 0.5 * np.pi, mag_samples)
      base_q = np.outer(q_range, unit_points).reshape(-1, 3)
      self.base_q = torch.from_numpy(base_q).float().cuda()
    elif base == 'mixed':
      # base translations
      t_range = 0.01
      tx_range, ty_range, tz_range = torch.linspace(-1*t_range, t_range, 16), torch.linspace(-1*t_range, t_range, 8), torch.linspace(-1*t_range, t_range, 16)
      tx, ty, tz = torch.meshgrid([tx_range, ty_range, tz_range]) 
      self.base_t = torch.stack((tx, ty, tz), dim=-1).reshape(-1, 3).cuda()
      #base orientations
      dir_samples = 256
      mag_samples = 8
      unit_points = fibonacci_sphere(dir_samples)
      q_range = np.linspace(0.001, 0.1 , mag_samples)
      # q_range = np.linspace(0.01, 0.5 * np.pi, mag_samples)
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
      a_xyz  = F.relu(self.rgb_fc_xyz(x))
      a_log_q = F.relu(self.rgb_fc_wpqr(x))
      if self.droprate > 0:
        a_xyz = self.dropout(a_xyz)
        a_log_q = self.dropout(a_log_q)
    else:
      # Depth feature extractor
      x = self.depth_feature_extractor(data)
      x = F.relu(x)
      a_xyz  = F.relu(self.depth_fc_xyz(x))
      a_log_q = F.relu(self.depth_fc_wpqr(x))
      if self.droprate > 0:
        a_xyz = self.dropout(a_xyz)
        a_log_q = self.dropout(a_log_q)
    xyz = torch.matmul(a_xyz, self.base_t)
    if self.bias is not None:
      xyz += self.bias
    log_q = torch.matmul(a_log_q, self.base_q)
    return torch.cat((xyz, log_q), dim=1)

class PoseNet_Basepose_Min(nn.Module):
  def __init__(self, feature_extractor, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0, bias=None):
    super(PoseNet_Basepose_Min, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    self.two_stream_mode = two_stream_mode
    fe_out_planes = feature_extractor.fc.in_features
    # bias setting
    if bias is not None:
      self.bias = torch.from_numpy(bias).cuda().float()
    else:
      self.bias = None
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
      # self.rgb_fc1_xyz  = nn.Linear(feat_dim, feat_dim / 2)
      # self.rgb_fc1_wpqr = nn.Linear(feat_dim, feat_dim / 2)
      # self.rgb_fc2_xyz  = nn.Linear(feat_dim / 2, feat_dim / 4)
      # self.rgb_fc2_wpqr = nn.Linear(feat_dim / 2, feat_dim / 4)
      # self.rgb_fc_xyz  = nn.Linear(feat_dim / 4, 6)
      # self.rgb_fc_wpqr = nn.Linear(feat_dim / 4, 6)
      self.rgb_fc_xyz  = nn.Linear(feat_dim, 6)
      self.rgb_fc_wpqr = nn.Linear(feat_dim, 6)
    elif self.two_stream_mode == 1:
      # Depth feature extractor
      # replace the last FC layer in feature extractor
      self.depth_feature_extractor = feature_extractor
      self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      # self.depth_fc1_xyz  = nn.Linear(feat_dim, feat_dim / 2)
      # self.depth_fc1_wpqr = nn.Linear(feat_dim, feat_dim / 2)
      # self.depth_fc2_xyz  = nn.Linear(feat_dim / 2, feat_dim / 4)
      # self.depth_fc2_wpqr = nn.Linear(feat_dim / 2, feat_dim / 4)
      # self.depth_fc_xyz  = nn.Linear(feat_dim / 4, 6)
      # self.depth_fc_wpqr = nn.Linear(feat_dim / 4, 6)
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
      # xyz  = F.relu(self.rgb_fc1_xyz(x))
      # wpqr = F.relu(self.rgb_fc1_wpqr(x))
      # xyz  = F.relu(self.rgb_fc2_xyz(xyz))
      # wpqr = F.relu(self.rgb_fc2_wpqr(wpqr))
      # a_xyz  = F.leaky_relu(self.rgb_fc_xyz(xyz))
      # a_log_q = F.leaky_relu(self.rgb_fc_wpqr(wpqr))
      a_xyz  = F.leaky_relu(self.rgb_fc_xyz(x))
      a_log_q = F.leaky_relu(self.rgb_fc_wpqr(x))
    else:
      # Depth feature extractor
      x = self.depth_feature_extractor(data)
      x = F.relu(x)
      if self.droprate > 0:
        x = self.dropout(x)
      # xyz  = F.relu(self.depth_fc1_xyz(x))
      # wpqr = F.relu(self.depth_fc1_wpqr(x))
      # xyz  = F.relu(self.depth_fc2_xyz(xyz))
      # wpqr = F.relu(self.depth_fc2_wpqr(wpqr))
      # a_xyz  = F.leaky_relu(self.depth_fc_xyz(xyz))
      # a_log_q = F.leaky_relu(self.depth_fc_wpqr(wpqr))
      a_xyz  = F.leaky_relu(self.depth_fc_xyz(x))
      a_log_q = F.leaky_relu(self.depth_fc_wpqr(x))
    xyz = torch.matmul(a_xyz, self.base_t)
    if self.bias is not None:
      xyz += self.bias
    log_q = torch.matmul(a_log_q, self.base_q)
    return torch.cat((xyz, log_q), dim=1)

class PoseNet_Basepose_Min_S2(nn.Module):
  def __init__(self, rgb_f_e, depth_f_e, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0, bias=None):
    super(PoseNet_Basepose_Min_S2, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    self.two_stream_mode = two_stream_mode
    fe_out_planes = rgb_f_e.fc.in_features
    # bias setting
    if bias is not None:
      self.bias = torch.from_numpy(bias).cuda().float()
    else:
      self.bias = None
    # unit bases
    base = np.array([[0., 0., 1.], [0., 1., 0.], [1., 0., 0.], [0., 0., -1.], [0., -1., 0.], [-1., 0., 0.]])
    self.base_t = torch.from_numpy(base).float().cuda()
    #base angles
    self.base_q = torch.from_numpy(base).float().cuda()

    if self.two_stream_mode == 5 or self.two_stream_mode == 6:
      # RGB feature extractor
      # replace the last FC layer in feature extractor
      self.rgb_feature_extractor = rgb_f_e
      self.rgb_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.rgb_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.rgb_fc_xyz  = nn.Linear(feat_dim, 6)
      self.rgb_fc_wpqr = nn.Linear(feat_dim, 6)

      # Depth feature extractor
      # replace the last FC layer in feature extractor
      self.depth_feature_extractor = depth_f_e
      self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.depth_fc_xyz  = nn.Linear(feat_dim, 6)
      self.depth_fc_wpqr = nn.Linear(feat_dim, 6)
    else:
      print 'two_stream_mode shoud be 5 (stage 2 of 2-stage training)or 6 (1-stage training) at the basepose_min 2nd stage'
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

  def forward(self, rgbs, depths):
    # RGB feature extractor
    x = self.rgb_feature_extractor(rgbs)
    x = F.relu(x)
    if self.droprate > 0:
        x = self.dropout(x)
    rgb_a_xyz  = F.leaky_relu(self.rgb_fc_xyz(x))
    rgb_a_log_q = F.leaky_relu(self.rgb_fc_wpqr(x))
    # Depth feature extractor
    y = self.depth_feature_extractor(depths)
    y = F.relu(y)
    if self.droprate > 0:
        y = self.dropout(y)
    depth_a_xyz  = F.leaky_relu(self.depth_fc_xyz(x))
    depth_a_log_q = F.leaky_relu(self.depth_fc_wpqr(x))
    # average fully-connected layer
    a_xyz = (rgb_a_xyz + depth_a_xyz) / 2.0
    a_log_q = (rgb_a_log_q + depth_a_log_q) / 2.0

    xyz = torch.matmul(a_xyz, self.base_t)
    log_q = torch.matmul(a_log_q, self.base_q)
    return torch.cat((xyz, log_q), dim=1)


class PoseNet_Basepose_S2(nn.Module):
  def __init__(self, rgb_f_e, depth_f_e, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0, t_range=0.01, q_range=0.01):
    super(PoseNet_Basepose_S2, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    self.two_stream_mode = two_stream_mode
    fe_out_planes = rgb_f_e.fc.in_features
    # base translations
    tx_range, ty_range, tz_range = torch.linspace(-1*t_range, t_range, 16), torch.linspace(-1*t_range, t_range, 8), torch.linspace(-1*t_range, t_range, 16)
    tx, ty, tz = torch.meshgrid([tx_range, ty_range, tz_range])
    self.base_t = torch.stack((tx, ty, tz), dim=-1).reshape(-1, 3).cuda()
    #base angles
    qx_range, qy_range, qz_range = torch.linspace(-1*q_range, q_range, 16), torch.linspace(-1*q_range, q_range, 8), torch.linspace(-1*q_range, q_range, 16)
    qx, qy, qz = torch.meshgrid([qx_range, qy_range, qz_range])
    self.base_q = torch.stack((qx, qy, qz), dim=-1).reshape(-1, 3).cuda()
    if self.two_stream_mode == 4:
      # RGB feature extractor
      # replace the last FC layer in feature extractor
      self.rgb_feature_extractor = rgb_f_e
      self.rgb_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.rgb_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.rgb_fc_xyz  = nn.Linear(feat_dim, feat_dim)
      self.rgb_fc_wpqr = nn.Linear(feat_dim, feat_dim)

      # Depth feature extractor
      # replace the last FC layer in feature extractor
      self.depth_feature_extractor = depth_f_e
      self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
      self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
      self.depth_fc_xyz  = nn.Linear(feat_dim, feat_dim)
      self.depth_fc_wpqr = nn.Linear(feat_dim, feat_dim)
    else:
      print 'two_stream_mode shoud be 4 at the basepose 2nd'
      raise NotImplementedError

    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [self.rgb_feature_extractor.fc, self.rgb_fc_xyz, self.rgb_fc_wpqr, self.depth_feature_extractor.fc, self.depth_fc_xyz, self.depth_fc_wpqr]
    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  def forward(self, rgbs, depths):
    # RGB feature extractor
    x = self.rgb_feature_extractor(rgbs)
    x = F.relu(x)
    if self.droprate > 0:
        x = self.dropout(x)
    rgb_a_xyz  = F.relu(self.rgb_fc_xyz(x))
    rgb_a_log_q = F.relu(self.rgb_fc_wpqr(x))
    # Depth feature extractor
    y = self.depth_feature_extractor(depths)
    y = F.relu(y)
    if self.droprate > 0:
        y = self.dropout(y)
    depth_a_xyz  = F.relu(self.depth_fc_xyz(x))
    depth_a_log_q = F.relu(self.depth_fc_wpqr(x))
    # average fully-connected layer
    a_xyz = (rgb_a_xyz + depth_a_xyz) / 2.0
    a_log_q = (rgb_a_log_q + depth_a_log_q) / 2.0

    xyz = torch.matmul(a_xyz, self.base_t)
    log_q = torch.matmul(a_log_q, self.base_q)
    return torch.cat((xyz, log_q), dim=1)



class PoseNet_S1(nn.Module):
  def __init__(self, feature_extractor, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=0):
    super(PoseNet_S1, self).__init__()
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
      xyz  = self.rgb_fc_xyz(x)
      log_q = self.rgb_fc_wpqr(x)
    else:
      # Depth feature extractor
      x = self.depth_feature_extractor(data)
      x = F.relu(x)
      if self.droprate > 0:
        x = self.dropout(x)
      xyz  = self.depth_fc_xyz(x)
      log_q = self.depth_fc_wpqr(x)
    return torch.cat((xyz, log_q), dim=1)

class PoseNet_S2(nn.Module):
  def __init__(self, rgb_f_e, depth_f_e, droprate=0, pretrained=True,
      feat_dim=2048, filter_nans=False, two_stream_mode=2):
    super(PoseNet_S2, self).__init__()
    self.droprate = droprate
    self.dropout = nn.Dropout(p=self.droprate)
    fe_out_planes = rgb_f_e.fc.in_features

    # RGB feature extractor
    # replace the last FC layer in feature extractor
    self.rgb_feature_extractor = rgb_f_e
    self.rgb_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    self.rgb_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
    # Depth feature extractor
    # replace the last FC layer in feature extractor
    self.depth_feature_extractor = depth_f_e
    self.depth_feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
    self.depth_feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
    '''
    self.fc_rgbd_fusion = nn.Linear(2*feat_dim, feat_dim)
    self.fc_xyz  = nn.Linear(feat_dim, 3)
    self.fc_wpqr = nn.Linear(feat_dim, 3)
    '''
    if two_stream_mode == 2:
      for p in self.parameters():
        p.requires_grad = False

    self.fc_xyz  = nn.Linear(2*feat_dim, 3)
    self.fc_wpqr = nn.Linear(2*feat_dim, 3)

    if filter_nans:
      self.fc_wpqr.register_backward_hook(hook=filter_hook)

    # initialize
    if pretrained:
      init_modules = [self.fc_xyz, self.fc_wpqr]#, self.fc_rgbd_fusion]

    else:
      init_modules = self.modules()

    for m in init_modules:
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)

  def forward(self, rgbs, depths):
    # RGB feature extractor
    x = self.rgb_feature_extractor(rgbs)
    x = F.relu(x)
    # Depth feature extractor
    y = self.depth_feature_extractor(depths)
    y = F.relu(y)
    # concatenate x and y
    x = torch.cat((x, y), dim=1)
    # x = self.fc_rgbd_fusion(x)
    xyz  = self.fc_xyz(x)
    log_q = self.fc_wpqr(x)

    return torch.cat((xyz, log_q), dim=1)

class MapNet(nn.Module):
  """
  Implements the MapNet model (green block in Fig. 2 of paper)
  """
  def __init__(self, mapnet, two_stream_mode=0):
    """
    :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
    of paper). Not to be confused with MapNet, the model!
    """
    super(MapNet, self).__init__()
    self.mapnet = mapnet
    self.two_stream_mode = two_stream_mode

  def forward(self, data, depths=None):
    """
    :param data: image blob (N x T x C x H x W) color images when mode=0, dn images when mode=1
    :param depths: image blob (N x T x C x H x W) depths is not None when mode>=2
    :return: pose nnd nd
     (N x T x 6)
    """
    s = data.size()
    data = data.view(-1, *s[2:])
    if self.two_stream_mode >= 2:
      if depths is None:
        raise Exception('Depth image cannot be None when using two stream CNN')
      depths = depths.view(-1, *s[2:])
      poses = self.mapnet(data, depths)
    else:
      poses = self.mapnet(data)
    
    poses = poses.view(s[0], s[1], -1)
    return poses

if __name__ == '__main__':
    main()