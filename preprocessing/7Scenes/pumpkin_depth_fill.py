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
import os
import os.path as osp
import matplotlib.pyplot as plt
import argparse
#
# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput * 1.0 / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

              # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    #print ('Solving system..')

    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    #print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth
    
    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
    return output
    
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
        d = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i))
                  for i in frame_idx]
        c_imgs.extend(c)
        d_imgs.extend(d)
            

    for i, (c_img, d_img) in enumerate(zip(c_imgs, d_imgs)):
        if i % 200 == 0:
            print 'Image {:d} / {:d}'.format(i, len(c_imgs))
        # c = np.array(Image.open(c_img))
        # d = np.array(Image.open(d_img).convert('I'))
        output_path = d_img.replace('depth', 'full_depth')
        if osp.exists(output_path):
            print output_path, 'exists'
            continue
        c = skimage.io.imread(c_img)
        d = skimage.io.imread(d_img, as_gray=True)
        d[d>3600] = 0
        d_fill = fill_depth_colorization(c, d).astype(np.uint16)
        
        # print d.dtype, d_fill.dtype
        skimage.io.imsave(output_path, d_fill)
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

