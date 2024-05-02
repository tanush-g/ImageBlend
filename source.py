import cv2
import numpy as np
from numpy.fft import ifft2, fft2
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.stats import norm
from scipy.signal import convolve2d
import math
import sys
import os
import scipy
import cv2

# Main function where the reconstruction is happening
def func(image1,image2,mask):
    # The mask is a binary mask where 1 represents that the second image 
    # would replace the first image at that position

    # Splitting channels
    (b1,g1,r1) = cv2.split(image1)
    r1 = r1.astype(float)
    g1 = g1.astype(float)
    b1 = b1.astype(float)
    (b2,g2,r2) = cv2.split(image2)
    r2 = r2.astype(float)
    g2 = g2.astype(float)
    b2 = b2.astype(float)
    (bm,gm,rm) = cv2.split(mask)
    rm = rm.astype(float)/255
    gm = gm.astype(float)/255
    bm = bm.astype(float)/255
    # Deciding the number of levels
    depth = int(math.floor(math.log(min(r1.shape[0], r1.shape[1]),2)))
    depth -= 4
    # Construct Laplacian Pyramids of both the images' channels
    lp_image1r = laplacian_pyramid(g_pyramid(r1, depth))
    lp_image1g = laplacian_pyramid(g_pyramid(g1, depth))
    lp_image1b = laplacian_pyramid(g_pyramid(b1, depth))
    lp_image2r = laplacian_pyramid(g_pyramid(r2, depth))
    lp_image2g = laplacian_pyramid(g_pyramid(g2, depth))
    lp_image2b = laplacian_pyramid(g_pyramid(b2, depth))
    # Blend and quantise the irregularities(if any)
    o_img_red = reconstruct(blend(lp_image2r,lp_image1r,g_pyramid(rm,depth)))
    o_img_red[o_img_red > 255] = 255
    o_img_red[o_img_red < 0] = 0
    o_img_red = o_img_red.astype(np.uint8)
    o_img_green = reconstruct(blend(lp_image2g,lp_image1g,g_pyramid(gm,depth)))
    o_img_green[o_img_green < 0] = 0
    o_img_green[o_img_green > 255] = 255
    o_img_green = o_img_green.astype(np.uint8)
    o_img_blue = reconstruct(blend(lp_image2b,lp_image1b,g_pyramid(bm,depth)))
    o_img_blue[o_img_blue < 0] = 0
    o_img_blue[o_img_blue > 255] = 255
    o_img_blue = o_img_blue.astype(np.uint8)
    # Add back the three channels
    tmp = []
    tmp.append(o_img_blue)
    tmp.append(o_img_green)
    tmp.append(o_img_red)
    result = np.zeros(image1.shape,dtype=image1.dtype)
    result = cv2.merge(tmp,result)
    cv2.imwrite('result.jpg',result)
    # Reduce dimensions for viewing purposes
    dimensions = (int(result.shape[1]*0.5), int(result.shape[0]*0.5))
    result = cv2.resize(result, dimensions, interpolation = cv2.INTER_AREA)
    cv2.imshow("Blended Image",result)
    # cv2.waitKey(0)
    return result

def expand_2(image):
    # Function for expanding by a factor of 4
    M = image.shape[0]
    N = image.shape[1]  
    o_img = np.zeros((2*M, 2*N), dtype=np.float64)
    # Used a symmetric 5x5 Kernel for blurring
    kernel = np.outer(np.array([0.05, 0.25, 0.4, 0.25, 0.05]),np.array([0.05, 0.25, 0.4, 0.25, 0.05]))
    o_img[::2,::2] = image[:,:]
    # return 4*scipy.signal.convolve2d(o_img,kernel,'same')
    # Convolution via FFT saves immense time
    return 4*np.abs(ifft2(np.multiply(fft2(o_img), fft2(kernel, s=o_img.shape))))
 
def laplacian_pyramid(g_pyramid):
    i = 0
    # Function for building a Laplacian Pyramid (Saving difference of each level)
    output = []
    while i < len(g_pyramid) - 1:
        X = g_pyramid[i]
        M, N = X.shape[0], X.shape[1]
        Y = expand_2(g_pyramid[i + 1])
        M_, N_ = Y.shape[0], Y.shape[1]
        if M_ > M:
                 Y = np.delete(Y,(-1),axis=0)
        if N_ > N:
                Y = np.delete(Y,(-1),axis=1)
        output.append(X - Y)
        i += 1
    output.append(g_pyramid.pop())
    return output

def blend(white, black, mask):
    res = []
    # Blending two laplacian pyramids using a mask
    for i in range(0, len(mask)):
        res.append((mask[i]*white[i]) + ((1 - mask[i])*black[i]))
    return res

def reconstruct(lp):
    M = lp[0].shape[0]
    N = lp[0].shape[1]
    # Function to build the image back from its laplacian pyramid
    output = np.zeros((M, N), dtype=np.float64)
    i = len(lp) - 1
    while i > 0:
        temp = lp[i - 1]
        lap = expand_2(lp[i])
        M, N = temp.shape[0], temp.shape[1]
        M_, N_ = lap.shape[0], lap.shape[1]
        if M_ > M:
                lap = np.delete(lap,(-1),axis=0)
        if N_ > N:
                lap = np.delete(lap,(-1),axis=1)
        lp.pop()
        tmp = lap + temp
        lp.pop()
        lp.append(tmp)
        output = tmp
        i -= 1
    return output

def g_pyramid(image, levels):
    output = []
    # Function for building the Gaussian Pyramid
    output.append(image)
    tmp = image
    for i in range(0,levels):
        kernel = np.outer(np.array([0.05, 0.25, 0.4, 0.25, 0.05]),np.array([0.05, 0.25, 0.4, 0.25, 0.05]))
        # o_img = scipy.signal.convolve2d(tmp,kernel,'same')
        # Convolution via FFT saves immense time
        o_img = np.abs(ifft2(np.multiply(fft2(tmp), fft2(kernel, s=tmp.shape))))
        # Reducing by a factor of 2
        out = o_img[::2,::2]
        tmp = out
        output.append(tmp)
    return output

def detect_face(copy_image):
    count_face = 0
    gray = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)
    # Taking min-neighbours to be 5 to just be sure
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    alldetect, face_radii, face_centers, center = [], [], [], []
    count_face = 0
    for (x, y, w, h) in faces:
        center = []
        face_radii.append(h / 2)
        # Draw a rectangle at the face co-ordinates
        copy_image = cv2.rectangle(copy_image,(x,y),(x + w,y + h),(255, 0, 0), 2)
        val_1 = x + (w / 2)
        val_2 = y + (w / 2)
        center.append(val_1)
        center.append(val_2)
        face_centers.append(center)
        count_face += 1  
    return face_radii, face_centers, count_face    

def feature_matching(img1,img2,face_centers,face_centers1,face_radii):
    orb = cv2.ORB_create()
    # Oriented BRIEF Keypoint Detection and Descriptor Extraction.
    # Function for keypoint and descriptor detecting
    first_image = img2
    second_image = img1
    kp1, des1 = orb.detectAndCompute(img1, None)
    msk = np.empty(first_image.shape)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # Initialising Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match the descriptors
    matches = bf.match(des1,des2)
    # Distance sorted matches
    matches = sorted(matches, key = lambda x:x.distance)
    plt.title('Feature Matching')
    f_match = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50], None,flags=0)
    plt.imshow(f_match)
    plt.show()
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    X = second_image.shape[0]
    Y = second_image.shape[1]
    im_out = cv2.warpPerspective(second_image, M, (Y, X)) 
    fccntr = []
    fccntr.append(face_centers1[0])
    fccntr = np.array(fccntr)
    dst = cv2.perspectiveTransform(fccntr[None,:,:],M)
    cv2.circle(msk,(int((dst[0,0,0])),int((dst[0,0,1]))),int((1.2*face_radii[0])),(255,255,255),-1)
    cv2.imshow('Mask', msk)
    # cv2.waitKey(0)
    func(first_image,im_out,msk)
    return M, mask

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    file_name_1 = input('Enter the name of the first image(with extension): ')
    file_name_2 = input('Enter the name of the second image(with extension): ')
    first_image = cv2.imread(file_name_1)
    # Creating Deep copies to ensure data integrity
    copy_image = deepcopy(first_image)
    second_image = cv2.imread(file_name_2)
    # Creating Deep copies to ensure data integrity
    copy_image1 = deepcopy(second_image)
    # Detecting Face for the first picture
    face_radii, face_centers, count_face = detect_face(copy_image)
    # Making a joined image for 
    both_images = np.concatenate((first_image, second_image), axis=1)
    # Similarly for the second picture
    Y, face_centers1, count_face1 = detect_face(copy_image1)
    dimensions = (int(both_images.shape[1]*0.5), int(both_images.shape[0]*0.5))
    face_radii.sort(key=dict(zip(face_radii, face_centers)).get)
    # Resize for display
    both_images = cv2.resize(both_images, dimensions, interpolation = cv2.INTER_AREA)
    face_centers = sorted(face_centers)
    cv2.imshow('Input Images', both_images)
    Y.sort(key=dict(zip(Y, face_centers1)).get)
    face_centers1 = sorted(face_centers1)
    # Trigger the call
    M, mask = feature_matching(second_image,first_image,face_centers,face_centers1,face_radii)
    # cv2.waitKey(0)
