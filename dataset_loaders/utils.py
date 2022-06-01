"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
from torchvision.datasets.folder import default_loader
from PIL import Image
import numpy as np

def load_image(filename, loader=default_loader):
  try:
    img = loader(filename)
  except IOError as e:
    print 'Could not load image {:s}, IOError: {:s}'.format(filename, e)
    return None
  except:
    print 'Could not load image {:s}, unexpected error'.format(filename)
    return None

  return img

def load_depth(filename):
  try:
    img = Image.open(filename).convert('I')
  except IOError as e:
    print 'Could not load image {:s}, IOError: {:s}'.format(filename, e)
    return None
  except:
    print 'Could not load image {:s}, unexpected error'.format(filename)
    return None

  return img

def load_point(filename):
  try:
    point = np.load(filename)
  except IOError as e:
    print 'Could not load point {:s}, IOError: {:s}'.format(filename, e)
    return None
  except:
    print 'Could not load point {:s}, unexpected error'.format(filename)
    return None

  return point