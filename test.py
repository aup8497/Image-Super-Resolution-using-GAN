import numpy as np
import scipy.misc
import cv2
import dlib
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from srgan import SRGAN

x = tf.placeholder(tf.float32, [None, 96, 96, 3])
is_training = tf.placeholder(tf.bool, [])

model = SRGAN(x, is_training, 16)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, 'check_point/latest')

img = cv2.imread('temp.jpg')
h, w = img.shape[:2]
detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)
if dets is None or len(dets) != 1:
    print("unsuitable")
    exit()
d = dets[0]
if d.left() < 0 or d.top() < 0 or d.right() > w or d.bottom() > h:
    print("unsuitable")
    exit()
face = img[d.top():d.bottom(), d.left():d.right()]
face = cv2.resize(face, (96, 96))
face = face / 127.5 - 1
input_ = np.zeros((16, 96, 96, 3))
input_[0] = face

mos, fake = sess.run(
    [model.downscaled, model.imitation],
    feed_dict={x: input_, is_training: False})

image1 = cv2.cvtColor(mos[0], cv2.COLOR_BGR2RGB);
image2 = cv2.cvtColor(fake[0], cv2.COLOR_BGR2RGB);
image3 = np.uint8((input_[0]+1)*127.5)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB);

print("saving the images")

scipy.misc.imsave('input.jpg',image1) # mos[0])
scipy.misc.imsave('generated.jpg',image2) #  fake[0])
scipy.misc.imsave('ground_truth.jpg',image3) #  input_[0])
