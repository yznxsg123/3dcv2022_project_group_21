"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Main training script for GeoMapNet
"""
import set_paths
from common.train import Trainer, load_state_dict_2stream, load_state_dict
from common.optimizer import Optimizer
from common.criterion import PoseNetCriterion, MapNetCriterion,\
  MapNetOnlineCriterion, GeoPoseNetCriterion, EvalCriterion
# from models.posenet import PoseNet, MapNet
from models.posenet_stages import PoseNet_S1, PoseNet_S2, PoseNet_Basepose, PoseNet_Basepose_Min, PoseNet_Basepose_S2, PoseNet_Basepose_Min_S2, MapNet
from dataset_loaders.composite import MF, MFOnline
import os.path as osp
import numpy as np
import argparse
import configparser
import json
import torch
from torch import nn
from torchvision import transforms, models
import random
import os

parser = argparse.ArgumentParser(description='Training script for PoseNet and'
                                             'MapNet variants')
parser.add_argument('--dataset', type=str, choices=('7Scenes', '12Scenes','RobotCar',  'TUM', 'AICL_NUIM'), help='Dataset', default='AICL_NUIM')
parser.add_argument('--scene', type=str, help='Scene name', default='livingroom')
parser.add_argument('--config_file', type=str, help='configuration file')
parser.add_argument('--model', choices=('posenet', 'mapnet', 'mapnet++', 'geoposenet'), help='Model to train', default='geoposenet')
parser.add_argument('--device', type=str, default='0',
  help='value to be set to $CUDA_VISIBLE_DEVICES')
parser.add_argument('--checkpoint', type=str, help='Checkpoint to resume from',
  default=None)
parser.add_argument('--learn_beta', action='store_true',
  help='Learn the weight of absolute pose loss')
parser.add_argument('--learn_gamma', action='store_true',
  help='Learn the weight of relative pose loss')
parser.add_argument('--learn_recon', action='store_true',
  help='Learn the weight of photometric loss and ssim loss')
parser.add_argument('--two_stream_mode', type=int, default=0, help='O: only RGB CNN, 1: only Depth CNN, 2: two stream fix two stream and train fc, 3:fine-tune all parameters')
parser.add_argument('--resume_optim', action='store_true',
  help='Resume optimization (only effective if a checkpoint is given')
parser.add_argument('--suffix', type=str, default='',
                    help='Experiment name suffix (as is)')
parser.add_argument('--gt_path', type=str, default='associate_gt.txt',
                    help='Ground truth path of TUM')
parser.add_argument('--d_suffix', type=str,
                    help='depth suffix of 7Scenes, depth for raw data full_depth for filling ')
parser.add_argument('--rgb_cp', type=str, help='Checkpoint for rgb stream to resume from at the second stage', default=None)
parser.add_argument('--depth_cp', type=str, help='Checkpoint for depth stream to resume from at the second stage', default=None)
parser.add_argument('--rgbd_cp', type=str, help='Checkpoint for two stream to resume from at the third stage', default=None)
parser.add_argument('--base', type=str, default='min', choices=('unfixed', 'cube', 'sphere', 'min', 'mixed'), help='type of base poses: unfixed / cube / sphere / min')
args = parser.parse_args()

settings = configparser.ConfigParser()
with open(args.config_file, 'r') as f:
  settings.read_file(f)
section = settings['optimization']
optim_config = {k: json.loads(v) for k,v in section.items() if k != 'opt'}
opt_method = section['opt']
lr = optim_config.pop('lr')
weight_decay = optim_config.pop('weight_decay')
# power = optim_config.pop('power')
power = None

section = settings['hyperparameters']
dropout = section.getfloat('dropout')
color_jitter = section.getfloat('color_jitter', 0)
sax = section.getfloat('sax')
saq = section.getfloat('saq')
if args.model.find('mapnet') >= 0 or args.model.find('geoposenet') >= 0:
  skip = section.getint('skip')
  real = section.getboolean('real')
  variable_skip = section.getboolean('variable_skip')
  srx = section.getfloat('srx')
  srq = section.getfloat('srq')
  steps = section.getint('steps')
if args.model.find('++') >= 0:
  vo_lib = section.get('vo_lib', 'orbslam')
  print 'Using {:s} VO'.format(vo_lib)
if args.model.find('geoposenet') >= 0:
  slp = section.getfloat('slp')
  sls = section.getfloat('sls')
  # ld = section.getfloat('ld')
  # lp = section.getfloat('lp')
  # ls = section.getfloat('ls')

section = settings['training']
seed = section.getint('seed')
max_epoch = section.getint('n_epochs')

# perseve reproducibility
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False  
torch.backends.cudnn.deterministic = True

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
  os.environ['CUDA_VISIBLE_DEVICES'] = args.device

# # read bias
data_dir = osp.join('..', 'data', args.dataset)
# bias_file = osp.join(data_dir, args.scene, 'position_mean.txt')
# bias = np.loadtxt(bias_file)
# print 'mean of position statistics using ' + bias_file
# print 'mean of position: {:s}'.format(bias)
bias = None

# model
# feature_extractor = models.resnet18(pretrained=True)
if args.two_stream_mode < 2:
  # feature_extractor = models.resnet34(pretrained=True)
  feature_extractor = models.resnet18(pretrained=True)
  if args.base == 'min':
    posenet = PoseNet_Basepose_Min(feature_extractor, droprate=dropout, pretrained=True, two_stream_mode=args.two_stream_mode, bias=bias)
  elif args.base == 'cube' or args.base == 'sphere' or args.base == 'mixed':
    posenet = PoseNet_Basepose(feature_extractor, droprate=dropout, pretrained=True, two_stream_mode=args.two_stream_mode, base=args.base, bias=bias)
  elif args.base == 'unfixed':
    posenet = PoseNet_S1(feature_extractor, droprate=dropout, pretrained=True, two_stream_mode=args.two_stream_mode)  
  else:
    raise Exception('Something wrong with the type of base poses: {:s}. Please check again.'.format(args.base))
  print 'Type of base poses: {:s}'.format(args.base)

else:
   # mode 2 and 3 and 4 and 5
  # rgb_f_e = models.resnet34(pretrained=False)
  # depth_f_e = models.resnet34(pretrained=False)
  rgb_f_e = models.resnet18(pretrained=False)
  depth_f_e = models.resnet18(pretrained=False)
  if args.two_stream_mode == 2:
    posenet = PoseNet_S2(rgb_f_e=rgb_f_e, depth_f_e=depth_f_e, droprate=dropout, pretrained=True, filter_nans=(args.model=='mapnet++'), two_stream_mode=args.two_stream_mode)
  elif args.two_stream_mode == 4:
    posenet = PoseNet_Basepose_S2(rgb_f_e=rgb_f_e, depth_f_e=depth_f_e, droprate=dropout, pretrained=True, filter_nans=(args.model=='mapnet++'), two_stream_mode=args.two_stream_mode)
  elif args.two_stream_mode == 5 or args.two_stream_mode == 6:
    posenet = PoseNet_Basepose_Min_S2(rgb_f_e=rgb_f_e, depth_f_e=depth_f_e, droprate=dropout, pretrained=True, filter_nans=(args.model=='mapnet++'), two_stream_mode=args.two_stream_mode)
  if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
  loc_func = None if torch.cuda.is_available() else lambda storage, loc: storage
  # load weights from the first stage
  if args.two_stream_mode == 2 or args.two_stream_mode == 4 or args.two_stream_mode == 5:
    if osp.isfile(args.rgb_cp) and osp.isfile(args.depth_cp):
      rgb_cp = torch.load(args.rgb_cp, map_location=loc_func)
      depth_cp = torch.load(args.depth_cp, map_location=loc_func)
      load_state_dict_2stream(posenet, rgb_cp['model_state_dict'], depth_cp['model_state_dict'])
      print 'Loaded weights from {:s} and {:s}'.format(args.rgb_cp, args.depth_cp)
    else:
      raise Exception('rgb checkpoint path: {:s} and depth checkpoint path: {:s} must exist at the begining of stage 2'.format(args.rgb_cp, args.depth_cp))
  '''
  # load weights from the second stage
  else:
    if osp.isfile(args.rgbd_cp):
      rgbd_cp = torch.load(args.rgbd_cp, map_location=loc_func)
      load_state_dict(posenet, rgbd_cp['model_state_dict'])
      print 'Loaded weights from {:s}'.format(args.rgbd_cp)
    else:
      raise Exception('rgbd checkpoint path: {:s} must exist at the begining of stage 3'.format(args.rgbd_cp))
  '''



if args.model == 'geoposenet':
  model = MapNet(mapnet=posenet, two_stream_mode=args.two_stream_mode)
elif args.model == 'posenet':
  model = posenet
elif args.model.find('mapnet') >= 0:
  model = MapNet(mapnet=posenet)
else:
  raise NotImplementedError

# loss function
if args.model == 'geoposenet':
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, slp=slp, sls=sls, learn_beta=args.learn_beta, learn_gamma=args.learn_gamma, learn_recon=args.learn_recon)#, ld=ld, lp=lp, ls=ls)
  if args.dataset == 'TUM':
    if args.scene == 'fr1': # TUM dataset
      K = torch.tensor([[517.3, 0, 318.6],
                        [0, 516.5, 255.3],
                        [0, 0, 1]]).float()
    else: # CoRBS dataset
      K = torch.tensor([[468.60, 0, 318.27],
                        [0, 468.61, 243.99],
                        [0, 0, 1]]).float()
    kwargs = dict(kwargs, depth_scale=5000, K=K)
  # ??????  not sure about K and depth scale of AICL_NUIM
  elif args.dataset == 'AICL_NUIM':
    K = torch.tensor([[481.20, 0, 319.5],
                      [0, -480, 239.5],
                      [0, 0, 1]]).float()
    kwargs = dict(kwargs, depth_scale=1000, K=K)
  train_criterion = GeoPoseNetCriterion(**kwargs)
  # val_criterion = GeoPoseNetCriterion()
  if args.dataset == 'RobotCar':
    pose_stats_file = osp.join('..', 'data', args.dataset, args.scene, 'pose_stats.txt')
    val_criterion = EvalCriterion(pose_stats_file=pose_stats_file)
  else:
    val_criterion = EvalCriterion()
elif args.model == 'posenet':
  train_criterion = PoseNetCriterion(sax=sax, saq=saq, learn_beta=args.learn_beta)
  val_criterion = PoseNetCriterion()
elif args.model.find('mapnet') >= 0:
  kwargs = dict(sax=sax, saq=saq, srx=srx, srq=srq, learn_beta=args.learn_beta,
                learn_gamma=args.learn_gamma)
  if args.model.find('++') >= 0:
    kwargs = dict(kwargs, gps_mode=(vo_lib=='gps') )
    train_criterion = MapNetOnlineCriterion(**kwargs)
    val_criterion = MapNetOnlineCriterion()
  else:
    train_criterion = MapNetCriterion(**kwargs)
    val_criterion = MapNetCriterion()
else:
  raise NotImplementedError

# optimizer
# filter parameters which don't require gradient
param_list = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]
# param_list = [{'params': model.parameters()}]

# for poly decay learning rate adjustment policy
# fc_ids = list(map(id, model.mapnet.feature_extractor.fc.parameters()))
# fc_ids.extend(list(map(id, model.mapnet.fc_xyz.parameters())))
# fc_ids.extend(list(map(id, model.mapnet.fc_wpqr.parameters())))
# fc_params = filter(lambda p: id(p) in fc_ids, model.parameters()) 
# block_params = filter(lambda p: id(p) not in fc_ids, model.parameters()) 
# param_list = [{'params': fc_params},
#               {'params': block_params, 'lr': 4 * lr }]
if args.learn_beta and hasattr(train_criterion, 'sax') and \
    hasattr(train_criterion, 'saq'):
  param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
if args.learn_gamma and hasattr(train_criterion, 'srx') and \
    hasattr(train_criterion, 'srq'):
  param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
if args.learn_recon and hasattr(train_criterion, 'slp') and \
    hasattr(train_criterion, 'sls'):
  param_list.append({'params': [train_criterion.slp, train_criterion.sls]})
optimizer = Optimizer(params=param_list, method=opt_method, base_lr=lr,
  weight_decay=weight_decay, power=power, max_epoch=max_epoch, **optim_config)



# transformers
'''
tforms = [transforms.Resize(256)]
if color_jitter > 0:
  assert color_jitter <= 1.0
  print 'Using ColorJitter data augmentation'
  tforms.append(transforms.ColorJitter(brightness=color_jitter,
    contrast=color_jitter, saturation=color_jitter, hue=0.5))
tforms.append(transforms.ToTensor())
tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
# tforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
data_transform = transforms.Compose(tforms)
'''

# crop_size_file = osp.join(data_dir, 'crop_size.txt')
# crop_size = tuple(np.loadtxt(crop_size_file).astype(np.int))

resize = (256, 341)
data_transform, depth_transform, dn_transform = None, None, None

scene_dn_scalar = {'fr1': 15000.0, 'desk': 20000.0, 'livingroom': 24924.0}

"""
No matter what two_stream_mode is, depth_transform is required for reconstruction loss using original depth image 
two_stream_mode == 0 : only data_transform for rgb CNN
two_stream_mode == 1 : only dn_transform for depth CNN
two_stream_mode == 2 or 3: both data_transform for rgb and dn_transform for depth CNN
"""


if args.two_stream_mode != 1:
  stats_file = osp.join(data_dir, args.scene, 'rgb_stats.txt')
  stats = np.loadtxt(stats_file)
  print 'rgb statistics using ' + stats_file
  print 'mean: {:s}\nvariance:{:s}'.format(stats[0], stats[1])

  if args.dataset == 'RobotCar':
    data_transform = transforms.Compose([
        transforms.Resize(resize),
        # transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=color_jitter,
                              contrast=color_jitter, saturation=color_jitter, hue=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))
      ])
  else:
    data_transform = transforms.Compose([
        transforms.Resize(resize),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))
      ])

depth_transform = transforms.Compose([
    transforms.Resize(resize),
    # transforms.ToTensor() won't normalize int16 array to [0, 1]
    transforms.ToTensor(),
    # convenient for division operation
	  transforms.Lambda(lambda x: x.float()),
    # important: only for 7Scenes
    transforms.Lambda(lambda x: torch.clamp(x, max=3600))
  ])
point_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

if args.two_stream_mode != 0:
  # only for TUM format
  if args.gt_path == 'associate_gt_fill_cmap.txt':
    depth_stats_fn = 'depth_cmap_stats.txt'
    dn_scalar = 221.0
  elif args.gt_path == 'associate_gt_fill.txt':
    depth_stats_fn = 'depth_stats.txt'
    dn_scalar = scene_dn_scalar[args.scene]
  # only for 7Scenes dataset
  if args.dataset == '7Scenes' or '12Scenes':
    depth_stats_fn = args.d_suffix + '_stats.txt'
    if args.d_suffix == 'full_d_cmap':
      dn_scalar = 221.0
    else:
      dn_scalar =  3600.0 # valid depth range of 7Scenes

  depth_stats = np.loadtxt( osp.join(data_dir, args.scene, depth_stats_fn) )
  print 'depth statistics using ' + depth_stats_fn
  print 'dn_scalar:', dn_scalar
  print 'mean: {:s}\nvariance:{:s}'.format(depth_stats[0], depth_stats[1])
  dn_transform = transforms.Compose([
      transforms.Resize(resize),
      # transforms.ToTensor() won't normalize int16 array to [0, 1]
      transforms.ToTensor(),
      # from [B, 1, H, W] to [B, C, H, W] and normalization
      # transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0).float() / dn_scalar[args.scene]),
      transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0).float() / dn_scalar),
      # important: only for 7Scenes
      transforms.Lambda(lambda x: torch.clamp(x, max=3600)),
      transforms.Normalize(mean=depth_stats[0], std=np.sqrt(depth_stats[1]))
    ])
  pn_transform = transforms.Compose([
      transforms.Lambda(lambda x: torch.from_numpy(x).float()),
      transforms.Normalize(mean=depth_stats[0], std=np.sqrt(depth_stats[1]))
    ])
target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())




# datasets
data_dir = osp.join('..', 'data', 'deepslam_data', args.dataset)
kwargs = dict(scene=args.scene, data_path=data_dir, transform=data_transform,
              target_transform=target_transform, seed=seed)
if args.model == 'geoposenet':
  if args.d_suffix:
    if args.d_suffix == 'cam_points' or args.d_suffix.find('scn_points') >= 0:
      print 'using 3D_point coordinates as input'
      kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
      variable_skip=variable_skip, depth_transform=point_transform, dn_transform=pn_transform, mode=args.two_stream_mode)
    else:
      kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
      variable_skip=variable_skip, depth_transform=depth_transform, dn_transform=dn_transform, mode=args.two_stream_mode)
  else:
    kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
      variable_skip=variable_skip, mode=args.two_stream_mode)
  if args.dataset == '7Scenes' or args.dataset == '12Scenes':
    train_set = MF(train=True, d_suffix=args.d_suffix, **kwargs)
    val_set = MF(train=False, d_suffix=args.d_suffix, **kwargs)
  elif args.dataset == 'TUM' or args.dataset == 'AICL_NUIM':
    train_set = MF(train=True, gt_path=args.gt_path, **kwargs)
    val_set = MF(train=False, gt_path=args.gt_path, **kwargs)
  elif args.dataset == 'RobotCar':
    train_set = MF(train=True, **kwargs)
    val_set = MF(train=False, **kwargs)
  else:
    raise NotImplementedError
elif args.model == 'posenet':
  if args.dataset == '7Scenes':
    from dataset_loaders.seven_scenes import SevenScenes
    train_set = SevenScenes(train=True, **kwargs)
    val_set = SevenScenes(train=False, **kwargs)
  elif args.dataset == 'RobotCar':
    from dataset_loaders.robotcar import RobotCar
    train_set = RobotCar(train=True, **kwargs)
    val_set = RobotCar(train=False, **kwargs)
  else:
    raise NotImplementedError
elif args.model.find('mapnet') >= 0:
  kwargs = dict(kwargs, dataset=args.dataset, skip=skip, steps=steps,
    variable_skip=variable_skip)
  if args.model.find('++') >= 0:
    train_set = MFOnline(vo_lib=vo_lib, gps_mode=(vo_lib=='gps'), **kwargs)
    val_set = None
  else:
    train_set = MF(train=True, real=real, **kwargs)
    val_set = MF(train=False, real=real, **kwargs)
else:
  raise NotImplementedError

# trainer
config_name = args.config_file.split('/')[-1]
config_name = config_name.split('.')[0]
experiment_name = '{:s}_{:s}_{:s}_{:s}'.format(args.dataset, args.scene,
  args.model, config_name)
if args.learn_beta:
  experiment_name = '{:s}_learn_beta'.format(experiment_name)
if args.learn_gamma:
  experiment_name = '{:s}_learn_gamma'.format(experiment_name)
if args.learn_recon:
  experiment_name = '{:s}_learn_recon'.format(experiment_name)
experiment_name += args.suffix
trainer = Trainer(model, optimizer, train_criterion, args.config_file,
                  experiment_name, train_set, val_set, device=args.device,
                  checkpoint_file=args.checkpoint,
                  resume_optim=args.resume_optim, val_criterion=val_criterion, two_stream_mode=args.two_stream_mode)
lstm = args.model == 'vidloc'
geopose = args.model == 'geoposenet'
trainer.train_val(lstm=lstm, geopose=geopose)
