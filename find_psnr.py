import cv2
import numpy as np 

img1 = cv2.imread("ground_truth.jpg")
img2 = cv2.imread("generated.jpg")
diff = img1-img2

cnt = 0
mse1 = 0
mse2 = 0
mse3 = 0
mse_avg = 0

print(diff.shape)

height = diff.shape[0]
width  = diff.shape[1]

mse = 0 
for y in range(height):
    for x in range(width):
        dif = img2[y][x] - img1[y][x]
        # print(dif**2)
        mse = mse + dif**2 / (height * width)

psnr = 10 * np.log10(pow(255, 2) / mse)
print("The PSNR between the images is ",np.average(psnr))

