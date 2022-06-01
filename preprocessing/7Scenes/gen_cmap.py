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
from PIL import Image
import os
import os.path as osp
import matplotlib.pyplot as plt
import argparse

    
def main(root_dir, val):
    if val:
        split_file = osp.join(root_dir, 'TestSplit.txt')
    else:
        split_file = osp.join(root_dir, 'TrainSplit.txt')

    with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]

    d_imgs = []
    c_imgs = []
    for seq in seqs:
        seq_dir = osp.join(root_dir, 'seq-{:02d}'.format(seq))
        p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if
                       n.find('pose') >= 0]
        frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
        c = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i))
                  for i in frame_idx]
        d = [osp.join(seq_dir, 'frame-{:06d}.full_depth.png'.format(i))
                  for i in frame_idx]
        c_imgs.extend(c)
        d_imgs.extend(d)
            

    for i, (c_img, d_img) in enumerate(zip(c_imgs, d_imgs)):
        if i % 200 == 0:
            print 'Image {:d} / {:d}'.format(i, len(c_imgs))
        # c = np.array(Image.open(c_img))
        # d = np.array(Image.open(d_img).convert('I'))
        d = skimage.io.imread(d_img, as_gray=True)
        output_path = d_img.replace('full_depth', 'full_d_cmap')
        cmap = plt.imsave(output_path, d, cmap='jet')
        # print d.dtype, d_fill.dtype
        # skimage.io.imsave(output_path, d_fill)
        # print np.mean(d_fill), np.mean(d)
        # plt.imshow(skimage.color.rgb2gray(c))
        # plt.show()
        # plt.imshow(np.concatenate((d, d_fill), axis=1), interpolation='nearest')
        # plt.show()

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

