import cv2
import numpy as np
import matplotlib.pyplot as plt


def gradient_simple(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = np.diff(img_gray, axis=1, append=0)
    grad_y = np.diff(img_gray, axis=0, append=0)
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_orient = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180
    return img_gray, grad_x, grad_y, grad, grad_orient


def show_gradients(img, img_gray, grad_x, grad_y, grad, grad_orient):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('Original Image')
    ax[0, 1].imshow(img_gray, cmap='gray')
    ax[0, 1].set_title('Gray Image')
    ax[1, 0].imshow(grad_x, cmap='gray')
    ax[1, 0].set_title('Gradient in direction X')
    ax[1, 1].imshow(grad_y, cmap='gray')
    ax[1, 1].set_title('Gradient in direction Y')
    ax[2, 0].imshow(grad, cmap='jet')
    ax[2, 0].set_title('Magnitude of the Gradient')
    ax[2, 1].imshow(grad_orient, cmap='jet')
    ax[2, 1].set_title('Orientation of the Gradient')
    plt.show()


def gradient_sobel(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad_orient = cv2.phase(np.array(grad_x, np.float32), np.array(grad_y, dtype=np.float32), angleInDegrees=True)
    return img_gray, grad_x, grad_y, grad, grad_orient


img_dog = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
imhead = img_dog.copy()

img_gray, grad_x, grad_y, grad, grad_orient = gradient_sobel(imhead)
show_gradients(imhead, img_gray, grad_x, grad_y, grad, grad_orient)


def gradient_canny(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(img_gray, 80, 200)
    return img_gray, edge


img_gray, edge = gradient_canny(img_dog)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
ax[0].imshow(img_dog)
ax[0].set_title('Original Image')
ax[1].imshow(edge, cmap='gray')
ax[1].set_title('Edges detected by Canny Filter')




import sys
import cv2 as cv
import numpy as np
import math


#hough lines transfomation:
default_file = 'dog.jpg'
filename = sys.argv[1] if len(sys.argv) > 1 else default_file

# Loads an image
src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

# Check if image is loaded fine
if src is None:
    print('Error opening image!')
else:
    print('Usage: hough_lines.py [image_name -- default ' + default_file + ']')

    dst = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()