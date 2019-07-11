import tensorflow as tf
import cv2

def load_vgg(sess, vgg_path):
    #helper.maybe_download_pretrained_vgg(vgg_path)
    model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    print('VGG16 Loaded')
    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name('image_input:0')
    layer_3 = graph.get_tensor_by_name('layer3_out:0')
    layer_4 = graph.get_tensor_by_name('layer4_out:0')
    layer_7 = graph.get_tensor_by_name('layer7_out:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
 
    return image_input, keep_prob, layer_3, layer_4, layer_7
    

def layers(vgglayer3_out, vgglayer4_out, vgglayer7_out, num_classes):


    ''' Better naming '''
    layer3, layer4, layer7 = vgglayer3_out, vgglayer4_out, vgglayer7_out

    '''
    You would conventionally apply a FC layer on top of layer 7.
    Do a 1x1 pooling and deduce fcn8
    TODO: a little more on 1x1 conv. What it does?
    '''

    fcn8 = tf.layers.conv2d(layer7, filters = num_classes, kernel_size = 1, name = 'fcn8')

    '''
    Upsample fcn8 to get fcn9.
    Match size of fcn9 to layer4 to aid adding a skip connection
    This can be done by inferring size from layer4
    '''
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters = layer4.get_shape().as_list()[-1],
                        padding = 'SAME', kernel_size = 4, strides = (2, 2), name = 'fcn9')
    
    '''Add Skip connection from layer4 to fcn9'''
    fcn9_skip_connected = tf.add(layer4, fcn9, name = 'fcn9__skip_connected')

    '''Upsample to fcn10, keeping in mind dimensions of layer3'''
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters = layer3.get_shape().as_list()[-1],
                        padding = 'SAME', kernel_size = 4, strides = (2, 2), name = 'fcn10')
    
    '''Add skip connection layer3+fcn10'''
    fcn10_skip_connected = tf.add(fcn10, layer3, name = 'fcn10_skip_connected')

    '''
    Upsample to match final image size
    TODO: Why is filters = num_classes?
    Found: the output map will contain two plots: for ROAD, and NOT ROAD
    '''
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
                                        kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")
    return fcn11

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

    '''Reshape tensors to 2d, pixel<->class mapping'''
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name = 'fcn_logits')

    gt_reshaped = tf.reshape(correct_label, (-1, num_classes), name = 'gt_reshaped')

    '''Calculate loss'''
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = gt_reshaped)
    loss_op = tf.reduce_mean(cross_entropy, name = 'fcn_loss')

    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_op)

    return logits, loss_op, train_op

def train_nn(sess, epochs, batch_size, get_batches_fn,
            train_op, loss_op, input_image, correct_label,
            keep_prob, learning_rate):
    
    keep_prob_value = 0.5
    learning_rate_value = 0.0001

    for epoch in range(epochs):
        '''Atleast initially'''
        total_loss = 0  
        for X_batch, gt_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([loss_op, train_op],
            feed_dict = {input_image: X_batch, correct_label: gt_batch,
            learning_rate: learning_rate_value, keep_prob: keep_prob_value})

            total_loss += loss

        print('Epoch {}...'.format(epoch+1))
        print('Loss {:.3f}'.format(total_loss))
        print()





