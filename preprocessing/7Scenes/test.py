# Original Matlab code https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
#
#
# Python port of depth filling code from NYU toolbox
# Speed needs to be improved
#
# Uses 'pypardiso' solver 
#
import scipy
import skimage
import numpy as np
from pypardiso import spsolve
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
import argparse

    
def main(root_dir):
    associate_path = osp.join(root_dir, 'associate_gt_fill.txt')
    rgbs = np.loadtxt(associate_path, dtype=str, usecols=1)
    depths = np.loadtxt(associate_path, dtype=str, usecols=3)
    c_imgs = [osp.join(root_dir, rgb) for rgb in rgbs]
    d_imgs = [osp.join(root_dir, depth) for depth in depths]
    for i, (c_img, d_img) in enumerate(zip(c_imgs, d_imgs)):
    	if i % 200 == 0:
            print 'Image {:d} / {:d}'.format(i, len(c_imgs))
        d = np.array(Image.open(d_img).convert('I')) / 65535.0
        output_path = d_img.replace('depth_fill', 'depth_colormap')
        plt.imsave(output_path, d, cmap='jet')
        

if __name__ == '__main__':
    # parse command line
    # root_dir = './datasets/TUM/fr1/rgbd_dataset_freiburg1_360'
    # parser = argparse.ArgumentParser(description='''
    # This script is depth encoder using jet colormap  
    # ''')
    # parser.add_argument('root_dir', help='scene\'s root directory')
    # args = parser.parse_args()
    
    # main(args.root_dir)
    fn1 = 'chess/seq-01/frame-000000.depth.png'
    fn2 = 'chess/seq-01/frame-000000.full_depth.png'
    img1 = np.array(Image.open(fn1).convert('I'))
    invalid = img1 >= 3600
    valid = img1 < 3600
    img1[invalid]=0
    img2 = np.array(Image.open(fn2).convert('I'))
    print np.max(img1), np.max(img2)
    print np.min(img1), np.min(img2)
    print np.mean(img1), np.mean(img2)
    diff = np.abs(img1-img2)
    num = np.sum(valid.astype(int))
    print num
    valid_mean = np.sum(diff[valid]) / num
    print valid_mean
    # output_path = fn1.replace('depth', 'cmp0')
    # plt.imsave(output_path, img1, cmap='jet')

