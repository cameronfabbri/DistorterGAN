import cPickle as pickle
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')
from tf_ops import *
import pix2pix
import data_ops

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--BATCH_SIZE',required=False,type=int,default=32,help='Batch size to use')
   parser.add_argument('--L1_WEIGHT', required=False,help='weight of L1 for combined loss',type=float,default=100.0)
   parser.add_argument('--NETWORK',   required=False,help='Architecture for the generator',default='pix2pix')
   parser.add_argument('--AUGMENT',   required=False,type=int,default=1,help='Whether or not to augment data')
   a = parser.parse_args()

   BATCH_SIZE = a.BATCH_SIZE
   L1_WEIGHT  = a.L1_WEIGHT
   NETWORK    = a.NETWORK
   AUGMENT    = bool(a.AUGMENT)

   EXPERIMENT_DIR = 'checkpoints'\
                    +'/NETWORK_'+NETWORK\
                    +'/L1_WEIGHT_'+str(L1_WEIGHT)\
                    +'/AUGMENT_'+str(AUGMENT)+'/'\

   IMAGES_DIR = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.makedirs(IMAGES_DIR)
   except: pass
   exit()

   # write all this info to a pickle file in the experiments directory
   exp_info = dict()
   exp_info['PRETRAIN_EPOCHS'] = PRETRAIN_EPOCHS
   exp_info['NETWORK']    = NETWORK
   exp_info['LOSS_METHOD']     = LOSS_METHOD
   exp_info['PRETRAIN_LR']     = PRETRAIN_LR
   exp_info['GAN_EPOCHS']      = GAN_EPOCHS
   exp_info['DATASET']         = DATASET
   exp_info['DATA_DIR']        = DATA_DIR
   exp_info['GAN_LR']          = GAN_LR
   exp_info['NUM_GPU']         = NUM_GPU
   exp_info['NUM_CRITIC']      = NUM_CRITIC
   exp_info['BATCH_SIZE']      = BATCH_SIZE
   exp_info['LOAD_MODEL']      = LOAD_MODEL
   exp_info['AUGMENT']          = AUGMENT
   exp_info['SIZE']            = SIZE
   exp_info['L1_WEIGHT']       = L1_WEIGHT
   exp_info['L2_WEIGHT']       = L2_WEIGHT
   exp_info['GAN_WEIGHT']      = GAN_WEIGHT
   exp_info['UPCONVS']          = UPCONVS
   exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'PRETRAIN_EPOCHS: ',PRETRAIN_EPOCHS
   print 'GAN_EPOCHS:      ',GAN_EPOCHS
   print 'NETWORK:    ',NETWORK
   print 'LOSS_METHOD:     ',LOSS_METHOD
   print 'PRETRAIN_LR:     ',PRETRAIN_LR
   print 'DATASET:         ',DATASET
   print 'DATA_DIR:        ',DATA_DIR
   print 'GAN_LR:          ',GAN_LR
   print 'NUM_GPU:         ',NUM_GPU
   print 'NUM_CRITIC:      ',NUM_CRITIC
   print 'LOAD_MODEL:      ',LOAD_MODEL
   print 'AUGMENT:          ',AUGMENT
   print 'SIZE:            ',SIZE
   print 'L1_WEIGHT:       ',L1_WEIGHT
   print 'L2_WEIGHT:       ',L2_WEIGHT
   print 'GAN_WEIGHT:      ',GAN_WEIGHT
   print

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # load data
   Data = data_ops.loadData(DATA_DIR, DATASET, BATCH_SIZE, jitter=AUGMENT, SIZE=SIZE)
   # number of training images
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   L_image   = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets

   lab_images = tf.concat([L_image, ab_image], axis=3)

   if NETWORK == 'pix2pix':
      # generated ab values from generator
      gen_ab = pix2pix.netG(L_image, UPCONVS)

   # D's decision on real images and fake images
   if LOSS_METHOD == 'energy':
      D_real, embeddings_real, decoded_real = pix2pix.energyNetD(L_image, ab_image, BATCH_SIZE)
      D_fake, embeddings_fake, decoded_fake = pix2pix.energyNetD(L_image, gen_ab, BATCH_SIZE, reuse=True)
   else:
      if NETWORK == 'pix2pix':
         D_real = pix2pix.netD(L_image, ab_images=ab_image)
         D_fake = pix2pix.netD(L_image, ab_images=gen_ab, reuse=True)

   genlab_images = tf.concat([L_image, gen_ab], axis=3)

   e = 1e-12
   if LOSS_METHOD == 'wasserstein':
      print 'Using Wasserstein loss'
      D_real = lrelu(D_real)
      D_fake = lrelu(D_fake)
      errD = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
      
      gen_loss_GAN = tf.reduce_mean(D_fake)

      # gradient penalty
      epsilon = tf.random_uniform([], 0.0, 1.0)
      x_hat = lab_images*epsilon + (1-epsilon)*genlab_images
      d_hat = pix2pix.netD(x_hat,reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty

      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1  = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2  = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      if L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using GAN loss, no L1 or L2'
         errG = gen_loss_GAN

   if LOSS_METHOD == 'least_squares':
      print 'Using least squares loss'
      # Least squares requires sigmoid activation on D
      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      
      #gen_loss_GAN = tf.reduce_mean(tf.square(errD_fake - 1))
      gen_loss_GAN = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1  = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2  = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      elif L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using GAN loss, no L1 or L2'
         errG = gen_loss_GAN
      #errD = tf.reduce_mean(tf.square(errD_real - 1) + tf.square(errD_fake))
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))

   if LOSS_METHOD == 'gan' or LOSS_METHOD == 'cnn':
      print 'Using original GAN loss'
      if LOSS_METHOD is not 'cnn':
         D_real = tf.nn.sigmoid(D_real)
         D_fake = tf.nn.sigmoid(D_fake)
         gen_loss_GAN = tf.reduce_mean(-tf.log(D_fake + e))
      else: gen_loss_GAN = 0.0
      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1  = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2  = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      if L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using GAN loss, no L1 or L2'
         #errD = errD + e
         errG = gen_loss_GAN
      errD = tf.reduce_mean(-(tf.log(D_real+e)+tf.log(1-D_fake+e)))
   
   if LOSS_METHOD == 'energy':
      print 'Using energy loss'
      margin = 80
      gen_loss_GAN = D_fake

      zero = tf.zeros_like(margin-D_fake)

      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1 = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG        = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2 = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG        = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      if L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using energy loss, no L1 or L2'
         errG = gen_loss_GAN
      errD = D_real + tf.maximum(zero, margin-D_fake)

   # tensorboard summaries
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   G_train_op = tf.train.AdamOptimizer(learning_rate=GAN_LR).minimize(errG, var_list=g_vars, global_step=global_step)
   D_train_op = tf.train.AdamOptimizer(learning_rate=GAN_LR).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=2)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
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
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)
   merged_summary_op = tf.summary.merge_all()
   start = time.time()
   
   epoch_num = step/(num_train/BATCH_SIZE)

   n_critic = NUM_CRITIC
      
   while epoch_num < GAN_EPOCHS:
      epoch_num = step/(num_train/BATCH_SIZE)
      s = time.time()
   
      for critic_itr in range(n_critic):
         sess.run(D_train_op)
         sess.run(clip_discriminator_var_op)
      sess.run(G_train_op)
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])

   summary_writer.add_summary(summary, step)
   if LOSS_METHOD != 'cnn' and step%10==0: print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-s
   else:
      if step%50==0:print 'epoch:',epoch_num,'step:',step,'loss:',loss,' time:',time.time()-s
   step += 1
   
   if step%500 == 0:
      print 'Saving model...'
      saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
      saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
      print 'Model saved\n'

   print 'Finished training', time.time()-start
   saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
   exit()
