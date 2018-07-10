import cPickle as pickle
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time
from skimage import io, color, transform
from tqdm import tqdm

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import data_ops

if __name__ == '__main__':

   BATCH_SIZE = 1

   if len(sys.argv) < 2:
      print 'You must provide an info.pkl file'
      exit()

   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)

   L1_WEIGHT = a['L1_WEIGHT']
   NETWORK   = a['NETWORK']
   AUGMENT   = a['AUGMENT']
   NOISE     = a['NOISE']

   EXPERIMENT_DIR = 'checkpoints'\
                    +'/NETWORK_'+NETWORK\
                    +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                    +'/NOISE_'+str(NOISE)\
                    +'/AUGMENT_'+str(AUGMENT)+'/'\

   DISTORTED_DIR    = EXPERIMENT_DIR+'distorted_gen/'
   NONDISTORTED_DIR = EXPERIMENT_DIR+'nondistorted_gen/'

   print
   try: os.makedirs(DISTORTED_DIR)
   except: pass
   try: os.makedirs(NONDISTORTED_DIR)
   except: pass

   print
   print 'L1_WEIGHT: ',L1_WEIGHT
   print 'NETWORK:   ',NETWORK
   print 'AUGMENT:   ',AUGMENT
   print 'NOISE:     ',NOISE
   print
   
   if NETWORK == 'pix2pix': from pix2pix import *
   if NETWORK == 'resnet': from resnet import *

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # load data
   distorted_paths, nondistorted_paths = data_ops.load_data()

   # number of training images
   num_train_d  = len(distorted_paths)
   num_train_nd = len(nondistorted_paths)

   # The gray 'lightness' channel in range [-1, 1]
   L_image = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 1), name='image_L')
   
   # possible noise to the generator - maybe it'll vary the output
   z = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 1), name='z')
   gen_ab = netG(L_image, z, NOISE)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   step = sess.run(global_step)
   total_num_train = num_train_d+num_train_nd

   for image_path in tqdm(nondistorted_paths):

      f_name = ntpath.basename(image_path)
      print image_path

      batchND_paths     = [image_path]
      batchND_L_images  = np.empty((BATCH_SIZE, 256, 256, 1), dtype=np.float64)
      batchReal         = np.empty((BATCH_SIZE, 256, 256, 3), dtype=np.float64)
      batch_z           = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 256,256,1]).astype(np.float32)

      j = 0
      for nd in batchND_paths:
         img_nd   = io.imread(nd)
         img_nd   = transform.resize(img_nd, (256,256,3))
         img_nd_  = color.rgb2lab(img_nd)
         img_nd_L = img_nd_[:,:,0]
         img_nd_L = np.expand_dims(img_nd_L, 2)
         img_nd_L = img_nd_L/50.0-1.
         batchND_L_images[j, ...] = img_nd_L
         batchReal[j,...] = img_nd
         j+=1

      generated_ab = sess.run(gen_ab, feed_dict={L_image:batchND_L_images, z:batch_z})
      generated_ab = generated_ab*128.

      for generated_ab, img_nd_L, real_img in zip(generated_ab, batchND_L_images, batchReal):
     
         img_nd_L = (img_nd_L+1)*50.

         rgb_out = np.concatenate([img_nd_L, generated_ab], axis=2)

         # clip to correct range
         rgb_out = np.clip(rgb_out, -128, 128)

         misc.imsave(NONDISTORTED_DIR+f_name, real_img)
         rgb_out  = color.lab2rgb(rgb_out)
         misc.imsave(DISTORTED_DIR+f_name, rgb_out)
         print 'Done'
         exit()
         counter += 1

         if counter == 5: break

