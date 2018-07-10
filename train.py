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

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import data_ops

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE',required=False,type=int,default=32,help='Batch size to use')
   parser.add_argument('--L1_WEIGHT', required=False,default=100.,type=float,help='Weight for L1 loss')
   parser.add_argument('--NETWORK',   required=False,help='Architecture for the generator',default='pix2pix')
   parser.add_argument('--AUGMENT',   required=False,type=int,default=0,help='Whether or not to augment data')
   parser.add_argument('--NOISE',     required=False,type=int,default=0,help='Whether or not to include noise to G')
   a = parser.parse_args()

   BATCH_SIZE = a.BATCH_SIZE
   L1_WEIGHT  = a.L1_WEIGHT
   NETWORK    = a.NETWORK
   AUGMENT    = bool(a.AUGMENT)
   NOISE      = bool(a.NOISE)

   EXPERIMENT_DIR = 'checkpoints'\
                    +'/NETWORK_'+NETWORK\
                    +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                    +'/NOISE_'+str(NOISE)\
                    +'/AUGMENT_'+str(AUGMENT)+'/'\

   IMAGES_DIR = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.makedirs(IMAGES_DIR)
   except: pass

   # write all this info to a pickle file in the experiments directory
   exp_info = dict()
   exp_info['NETWORK']   = NETWORK
   exp_info['AUGMENT']   = AUGMENT
   exp_info['L1_WEIGHT'] = L1_WEIGHT
   exp_info['NOISE']     = NOISE
   exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'NETWORK: ',NETWORK
   print 'AUGMENT: ',AUGMENT
   print 'NOISE:   ',NOISE
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

   '''
      So we can have either just 1 L channel or two, one coming from the distorted
      dataset and one coming from the nondistorted dataset.
   '''
   # The gray 'lightness' channel in range [-1, 1]
   L_image = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 1), name='image_L')
   
   # The ab color channels in [-1, 1] range
   ab_image = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 2), name='ab_image')

   # full image
   lab_images = tf.concat([L_image, ab_image], axis=3)

   # possible noise to the generator - maybe it'll vary the output
   z = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 256, 256, 1), name='z')

   '''
      generated ab values from generator given the lightness.
      D_real is giving the discriminator real distorted images.
      D_fake is giving the discriminator generated distorted images.
   '''
   gen_ab = netG(L_image, z, NOISE)
   D_real = netD(L_image, ab_image)
   D_fake = netD(L_image, gen_ab, reuse=True)

   genlab_images = tf.concat([L_image, gen_ab], axis=3)

   e = 1e-12
   errD = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
   errG = -tf.reduce_mean(D_fake)

   if L1_WEIGHT > 0.0:
      l1_loss = tf.reduce_mean(tf.abs(ab_image-gen_ab))
      errG += L1_WEIGHT*l1_loss

   # gradient penalty
   epsilon = tf.random_uniform([], 0.0, 1.0)
   x_hat = ab_image*epsilon + (1-epsilon)*gen_ab
   d_hat = netD(L_image,x_hat,reuse=True)
   gradients = tf.gradients(d_hat, x_hat)[0]
   slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
   gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
   errD += gradient_penalty

   # tensorboard summaries
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion
   step = sess.run(global_step)
   merged_summary_op = tf.summary.merge_all()
   
   total_num_train = num_train_d+num_train_nd

   epoch_num = step/(total_num_train/BATCH_SIZE)
   n_critic  = 5

   # have to remember the number of images is different for distorted and nondistorted
        
   # L_chan: black and white with input range [0, 100]
   # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
   # [0, 100] => [-1, 1],  ~[-128, 128] => [-1, 1]

   while epoch_num < 100:
      
      epoch_num = step/(total_num_train/BATCH_SIZE)
      
      for critic_itr in range(n_critic):
         
         idx_d  = np.random.choice(np.arange(num_train_d), BATCH_SIZE, replace=False)
         #idx_nd = np.random.choice(np.arange(num_train_nd), BATCH_SIZE, replace=False)
         batchD_paths  = distorted_paths[idx_d]
         #batchND_paths = nondistorted_paths[idx_nd]

         batchD_L_images  = np.empty((BATCH_SIZE, 256, 256, 1), dtype=np.float64)
         batchD_ab_images = np.empty((BATCH_SIZE, 256, 256, 2), dtype=np.float64)

         #batchND_L_images  = np.empty((BATCH_SIZE, 256, 256, 1), dtype=np.float32)
         #batchND_ab_images = np.empty((BATCH_SIZE, 256, 256, 2), dtype=np.float32)

         batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 256,256,1]).astype(np.float32)

         i = 0
         #for d,nd in zip(batchD_paths, batchND_paths):
         for d in batchD_paths:
            img_d    = io.imread(d)
            img_d    = transform.resize(img_d, (256,256,3))
            img_d    = color.rgb2lab(img_d)
            img_d_L  = img_d[:,:,0]
            img_d_L  = np.expand_dims(img_d_L, 2)
            img_d_L  = img_d_L/50.0-1.
            img_d_ab = img_d[:,:,1:]
            img_d_ab = img_d_ab/128.

            '''
            img_nd    = io.imread(nd)
            img_nd    = transform.resize(img_nd, (256,256,3))
            img_nd    = color.rgb2lab(img_nd)
            img_nd_L  = img_nd[:,:,0]
            img_nd_L  = np.expand_dims(img_nd_L, 2)
            img_nd_L  = img_nd_L/50.0-1
            img_nd_ab = img_nd[:,:,1:]
            img_nd_ab = img_nd_ab/128

            batchND_L_images[i, ...] = img_nd_L
            batchND_ab_images[i, ...] = img_nd_ab
            '''

            batchD_L_images[i, ...] = img_d_L
            batchD_ab_images[i, ...] = img_d_ab
            i+=1

         sess.run(D_train_op, feed_dict={L_image:batchD_L_images, ab_image:batchD_ab_images,z:batch_z})
      sess.run(G_train_op, feed_dict={L_image:batchD_L_images, ab_image:batchD_ab_images,z:batch_z})

      # get loss without gradient update
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={L_image:batchD_L_images, ab_image:batchD_ab_images,z:batch_z})
      print 'epoch:',epoch_num,'step:',step,'D_loss:',D_loss,'G_loss:',G_loss

      step += 1

      summary_writer.add_summary(summary, step)
      
      if step%100 == 0:
         print 'Saving model...'
         saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'

         # test out on some nondistorted images
         idx_nd            = np.random.choice(np.arange(num_train_nd), BATCH_SIZE, replace=False)
         batchND_paths     = nondistorted_paths[idx_nd]
         batchND_L_images  = np.empty((BATCH_SIZE, 256, 256, 1), dtype=np.float64)
         batchND_ab_images = np.empty((BATCH_SIZE, 256, 256, 2), dtype=np.float64)
            
         batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 256,256,1]).astype(np.float32)

         j = 0
         for nd in batchND_paths:
            img_nd    = io.imread(nd)
            img_nd    = transform.resize(img_nd, (256,256,3))
            img_nd    = color.rgb2lab(img_nd)
            img_nd_L  = img_nd[:,:,0]
            img_nd_L  = np.expand_dims(img_nd_L, 2)
            img_nd_L  = img_nd_L/50.0-1.
            img_nd_ab = img_nd[:,:,1:]
            img_nd_ab = img_nd_ab/128.
            batchND_L_images[j, ...] = img_nd_L
            batchND_ab_images[j, ...] = img_nd_ab
            j+=1

         generated_ab = sess.run(gen_ab, feed_dict={L_image:batchND_L_images, z:batch_z})

         counter = 0
         for real_ab, ab_gen_, real_L in zip(batchND_ab_images, generated_ab, batchND_L_images):

            # put back to scale
            real_L  = (real_L+1)*50.
            real_ab = real_ab*128.
            ab_gen_ = ab_gen_*128.

            real_img = np.concatenate([real_L, real_ab], axis=2)
            rgb_out = np.concatenate([real_L, ab_gen_], axis=2)

            # clip to correct range
            rgb_out = np.clip(rgb_out, -128, 128)

            real_img = color.lab2rgb(real_img)
            misc.imsave(IMAGES_DIR+str(step)+'_'+str(counter)+'_real.png', real_img)
            rgb_out  = color.lab2rgb(rgb_out)
            misc.imsave(IMAGES_DIR+str(step)+'_'+str(counter)+'_gen.png', rgb_out)

            counter += 1

            if counter == 5: break

