'''
   Takes a directory of images and plots them.
   Can also use features from some pretrained
   network.
'''
import os
import sys
import time
import fnmatch
import argparse
import numpy as np
import cPickle as pickle
import scipy.misc as misc
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

def getPaths(data_dir):
   exts = ['*.png','*.jpg','*.JPEG','*.JPG','*.PNG']
   image_paths = []
   for pattern in exts:
      for d, s, fList in os.walk(data_dir):
         for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
               fname_ = os.path.join(d,filename)
               image_paths.append(fname_)
   return image_paths

def plotImage(x, y, im):
   bb = Bbox.from_bounds(x,y,1,1)  
   bb2 = TransformedBbox(bb,ax.transData)
   bbox_image = BboxImage(bb2,
                          norm = None,
                          origin=None,
                          clip_on=False)
   bbox_image.set_data(im)
   ax.add_artist(bbox_image)

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--data_dir',required=True,type=str,help='Directory containing images to use')
   parser.add_argument('--feature_file',required=False,type=str,help='Features to use',default='pixels')
   a = parser.parse_args()
   
   data_dir     = a.data_dir
   feature_file = a.feature_file

   paths = getPaths(data_dir)
   # array containing pixels or features
   X = []

   # if using features from a CNN, load them here
   print 'Loading...'
   if feature_file is not 'pixels':
      pkl_file = open(feature_file, 'rb')
      features = pickle.load(pkl_file)
      paths = []
      for p, feature in features.iteritems():
         paths.append(p)
         X.append(feature)
   else:
      print 'Using pixels as features'
      for p in paths:
         X.append(misc.imresize(misc.imread(p), (256,256,3)).flatten())

   X = np.asarray(X)
   print 'Done\n'

   s = time.time()
   print 'fitting tsne'
   X_embedded = TSNE(n_components=2).fit_transform(X)
   print '\n Done after',time.time()-s

   '''
      Now want to plot the actual images (small version) on a 2d plot.
      We'll get the (x,y) location from tsne above
   '''
   x = []
   y = []
   imgs = []

   for embedding, path in zip(X_embedded, paths):
      imgs.append(misc.imresize(misc.imread(path), (512,512,3)))
      x.append(embedding[0])
      y.append(embedding[1])

   fig = plt.figure()
   ax  = fig.add_subplot(111)


# plot the images on their x,y points
   for x_,y_,img in zip(x,y,imgs):
      plotImage(x_,y_,img)

   ax.set_ylim(np.min(y),np.max(y))
   ax.set_xlim(np.min(x),np.max(x))

   plt.savefig('tsne_out.png', dpi = 3000)
   #plt.show()
