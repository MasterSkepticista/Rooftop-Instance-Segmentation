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
EPOCHS = 1
BATCH_SIZE = 1
DROPOUT = 0.75
image_shape = (600, 600)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Placeholders
correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], NUM_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Directories
root = 'data'
rgb_dir = root + '/train/image'
gt_dir = root + '/train/label'
vgg_path = 'vgg/vgg'
runs_dir = './runs'

# helper.maybe_download_pretrained_vgg(vgg_path)
'''
Get image by concatenating <dir_name> and <file_from_list>
Example: image = cv2.imread(rgb_dir + rgb_list[0], 1) to read first image from RGB
'''


get_batches_fn = helper.gen_batch_function(root, image_shape)

print('Object ready')
for X, y in get_batches_fn(2):
        '''
        Snippet to view X and y samples
        '''

        gt = 255 * np.stack([y[0][:,:,0], y[0][:,:,0], y[0][:,:,0]], axis = -1)
        
        fig, axes = plt.subplots(2)
        axes[0].imshow(X[0])
        axes[1].imshow(gt)
        plt.show()
        '''
        End Snippet here
        '''