import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os

#####################################################
# reading images into numpy array from a folder
#####################################################
train_folder = 'processed_images_data/sample/'
test_folder = 'processed_images_data/sample/'



def downscale(x):
    K = 4
    arr = np.zeros([K, K, 3, 3])
    arr[:, :, 0, 0] = 1.0 / K ** 2
    arr[:, :, 1, 1] = 1.0 / K ** 2
    arr[:, :, 2, 2] = 1.0 / K ** 2
    weight = tf.constant(arr, dtype=tf.float32)
    x = tf.cast(x, tf.float32)		# we have to cast this to float
    downscaled = tf.nn.conv2d(
        x, weight, strides=[1, K, K, 1], padding='SAME')
    return downscaled

# NOTE : cv2.imread will directly give numpy array
for fname in os.listdir(train_folder):
    image=cv2.imread(train_folder + fname)
    # data = np.append(data, image)
    z = image
    print(z)
    plt.imshow(downscale([1,z[0],z[1],z[2]]))
    plt.show()
    # print(type(image))
    # i=i+1

## NOTE : cv2.imread will directly give numpy array
# for fname in os.listdir(test_folder):
#     image=cv2.imread(test_folder + fname)
#     # data = np.append(data, image)
#     print(type(image))
#     # i=i+1


