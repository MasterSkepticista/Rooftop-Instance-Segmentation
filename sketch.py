import os
import tensorflow as tf
import numpy as np
import cv2
import helper_functions
import helper
import tqdm
import matplotlib.pyplot as plt
# Define constants
NUM_CLASSES = 2     
EPOCHS = 17
BATCH_SIZE = 2
DROPOUT = 0.75
image_shape = (3584, 3584)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Placeholders
correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], NUM_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Directories
root = 'data'
rgb_dir = root + '/train/image'
gt_dir = root + '/train/label'


vgg_path = 'vgg' # When downloading VGG model
# vgg_path = 'vgg/vgg' # Uncomment this when you have downloaded 
runs_dir = './runs'

# Comment this when downloaded VGG model
helper.maybe_download_pretrained_vgg(vgg_path)
'''
Get image by concatenating <dir_name> and <file_from_list>
Example: image = cv2.imread(rgb_dir + rgb_list[0], 1) to read first image from RGB
'''


get_batches_fn = helper.gen_batch_function(root, image_shape)
with tf.Session() as sess:

        '''
        Get the required layers from pretrained model
        '''
        input_tensor, keep_prob, layer3, layer4, layer7 = helper_functions.load_vgg(sess, vgg_path)
        
        '''
        Obtain upsampled output layer from these three, skip connected layers
        returns: fcn11
        '''
        model_output = helper_functions.layers(layer3, layer4, layer7, NUM_CLASSES)

        '''
        Compile model and evaluate loss
        Then Train
        '''

        logits, loss_op, train_op = helper_functions.optimize(model_output, correct_label, learning_rate, NUM_CLASSES)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("Model Built")
        helper_functions.train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn,
                train_op, loss_op, input_tensor, correct_label, keep_prob, learning_rate)
        print("Finished training")

        helper.save_inference_samples(runs_dir, 'data/test', sess, image_shape, logits, keep_prob, input_tensor)
