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


    
def main(root_dir, val, depth_scale=1000):
    if val:
        split_file = osp.join(root_dir, 'TestSplit.txt')
    else:
        split_file = osp.join(root_dir, 'TrainSplit.txt')

    with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

    fulld_imgs = []
    depth_imgs = []
    for seq in seqs:
        seq_dir = osp.join(root_dir, 'seq-{:02d}'.format(seq))
        p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('pose') >= 0]
        frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
        fulld = [osp.join(seq_dir, 'frame-{:06d}.full_depth.png'.format(i))
                  for i in frame_idx]
        fulld_imgs.extend(fulld)
        depth = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i))
                  for i in frame_idx]
        depth_imgs.extend(depth)
            
    n_valid = 0
    valid_value = 0
    value = 0
    n_strict_valid = 0
    strict_value = 0
    maximum, minimum = 0, 65535
    for i, fulld_img in enumerate(fulld_imgs):
        # if i % 200 == 0:
        #     print( 'Image {:d} / {:d}'.format(i, len(fulld_imgs))) 
        fulld = np.array(Image.open(fulld_img).convert('I'))
        # if(i==0):
        #     print(fulld.shape)
        # d[d>3600] = 0
        valid = (fulld > 0) & (fulld < 65535)
        n_valid += np.sum(valid)
        valid_value += np.sum(fulld[valid] / 10.0)
        value += np.sum(fulld / 10.0)
        strict_valid = (fulld > 0 & (fulld < 3600)
        n_strict_valid += np.sum(strict_valid)
        strict_value += np.sum(fulld[strict_valid] / 10.0)
        maximum = fulld.max() if fulld.max() > maximum else maximum
        minimum = fulld.min() if fulld.min() < minimum else minimum

    N = len(fulld_imgs) * 640*480
    # print 'The number of valid depth is ', n_valid
    print ('----------------full_depth----------------------')
    print (maximum, minimum)
    print ('Mean of distance is ', value / N, ' cm')
    print ('The ratio of valid depth and all depth is ', n_valid * 100.0 / N)
    print ('Mean of valid distance is ', valid_value / n_valid, ' cm')
    print ('The ratio of strict valid depth and all depth is ', n_strict_valid * 100.0 / N)
    print ('Mean of strict valid distance is ', strict_value / n_strict_valid, ' cm')

    n_valid = 0
    valid_value = 0
    value = 0
    n_strict_valid = 0
    strict_value = 0
    maximum, minimum = 0, 65535
    for i, depth_img in enumerate(depth_imgs):
        # if i % 200 == 0:
        #     print( 'Image {:d} / {:d}'.format(i, len(depth_imgs))) 
        depth = np.array(Image.open(depth_img).convert('I'))
        # d[d>3600] = 0
        valid = (depth > 0) & (depth < 65535)
        n_valid += np.sum(valid)
        valid_value += np.sum(depth[valid] / 10.0)
        value += np.sum(depth / 10.0)
        strict_valid = (depth > 0) & (depth < 3600)
        n_strict_valid += np.sum(strict_valid)
        strict_value += np.sum(depth[strict_valid] / 10.0)
        maximum = depth.max() if depth.max() > maximum and depth.max()!= 65535 else maximum
        minimum = depth.min() if depth.min() < minimum else minimum
    # print 'The number of valid depth is ', n_valid
    print ('----------------depth----------------------')
    print (maximum, minimum)
    print ('Mean of distance is ', value / N, ' cm')
    print ('The ratio of valid depth and all depth is ', n_valid * 100.0 / N)
    print ('Mean of valid distance is ', valid_value / n_valid, ' cm')
    print ('The ratio of strict valid depth and all depth is ', n_strict_valid * 100.0 / N)
    print ('Mean of strict valid distance is ', strict_value / n_strict_valid, ' cm')



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

