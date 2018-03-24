import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from srgan import SRGAN
import load 

image_dim = 96
num_Images = 3
i=0

learning_rate = 1e-3
batch_size = 10
vgg_model = 'check_point/latest'

def train():
    x = tf.placeholder(tf.float32, [None, 96, 96, 3]) # this is the image
    is_training = tf.placeholder(tf.bool, [])

    model = SRGAN(x, is_training, batch_size)
    sess = tf.Session()
    with tf.variable_scope('srgan'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
# just defining here, will be executed later
    # generator loss 
    g_train_op = opt.minimize(
        model.g_loss, global_step=global_step, var_list=model.g_variables)
    # discriminator loss
    d_train_op = opt.minimize(
        model.d_loss, global_step=global_step, var_list=model.d_variables)
    
    init = tf.global_variables_initializer() 
    # print(init)
    sess.run(init)

    print("restoring the model")
    # Restore the VGG-19 network
    var = tf.global_variables()
    vgg_var = [var_ for var_ in var if "vgg19" in var_.name]
    saver = tf.train.Saver(vgg_var)
    saver.restore(sess, vgg_model)

    print("restoring the SRGAN model")
    # Restore the SRGAN network
    # if tf.train.get_checkpoint_state('check_point/'):
    #     saver = tf.train.Saver()
    #     saver.restore(sess, 'check_point/latest')

    print("loading data")
    # # Load the data
    x_train, x_test = load.load()

    # Train the SRGAN model

    n_iter = int(len(x_train) / batch_size)
    while True:
        epoch = int(sess.run(global_step) / n_iter / 2) + 1
        print('epoch:', epoch)
        np.random.shuffle(x_train)
        for i in tqdm(range(10)):
            x_batch = normalize(x_train[i*batch_size:(i+1)*batch_size])
            sess.run(
                [g_train_op, d_train_op],
                feed_dict={x: x_batch, is_training: True})

        # Validate
        raw = normalize(x_test[:batch_size])
        mos, fake = sess.run(
            [model.downscaled, model.imitation],
            feed_dict={x: raw, is_training: False})

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess, 'check_point/latest', write_meta_graph=False)


def normalize(images):
    return np.array([image/127.5-1 for image in images])

if __name__ == '__main__':
    train()

