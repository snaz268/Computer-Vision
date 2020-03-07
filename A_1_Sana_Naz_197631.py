# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:26:02 2020

@author: sanan
"""
import matplotlib.pyplot as plt
import cv2
import numpy as np

def draw_keypoints(kp, img):
    sift_img = cv2.drawKeypoints(img, kp, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return sift_img

def matcher(img1, img2, kp1, des1, kp2, des2, text):
  #kp1, des1 = sift(img1)
  kp_img1 = draw_keypoints(kp1, img1)

  #kp2,des2 = sift(img2)
  kp_img2 = draw_keypoints(kp2, img2)

  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1, des2, k=2)
  # Apply ratio test
  good = []
  for m,n in matches:
      if m.distance < 0.75*n.distance:
          good.append([m])
  # cv.drawMatchesKnn expects list of lists as matches.
  bf_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

  fig = plt.figure(figsize=(11,11))
  a = fig.add_subplot(2, 2, 1)
  a.set_title("Image 1")
  plt.imshow(img1)

  a = fig.add_subplot(2, 2, 2)
  a.set_title(text)
  plt.imshow(kp_img1)

  a = fig.add_subplot(2, 2, 3)
  a.set_title("Image 2")
  plt.imshow(img2)

  a = fig.add_subplot(2, 2, 4)
  a.set_title(text)
  plt.imshow(kp_img2)

  fig2 = plt.figure(figsize=(12,12))
  a = fig2.add_subplot(1, 1, 1)
  a.set_title("BF Matcher")
  plt.imshow(bf_img)

def sift(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    matcher(img1, img2, kp1, des1, kp2, des2, "SIFT Keypoints")
  
def surf(img1, img2):
    surf = cv2.xfeatures2d.SURF_create(400)
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)
    matcher(img1, img2, kp1, des1, kp2, des2, "SURF Keypoints")  
  
def brief(img1, img2):
# Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    kp1 = star.detect(img1,None)
    kp2 = star.detect(img2,None)
    # compute the descriptors with BRIEF
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)

    matcher(img1, img2, kp1, des1, kp2, des2, "BREIF Keypoints")
    
def orb(img1, img2):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp1 = orb.detect(img1,None)
    kp2 = orb.detect(img2,None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    matcher(img1, img2, kp1, des1, kp2, des2, "ORB Keypoints")
    
img1 = cv2.imread('D:\\NUST\\8th Semester\\Computer Vision\\Labs\\Assignment 1\\book.jpg')
img2 = cv2.imread('D:\\NUST\\8th Semester\\Computer Vision\\Labs\\Assignment 1\\book_person_holding.jpg')
sift(img1, img2)
surf(img1, img2)
brief(img1, img2)
orb(img1, img2)

img1 = cv2.imread('D:\\NUST\\8th Semester\\Computer Vision\\Labs\\Assignment 1\\roma_1.jpg')
img2 = cv2.imread('D:\\NUST\\8th Semester\\Computer Vision\\Labs\\Assignment 1\\roma_2.jpg')
sift(img1, img2)
surf(img1, img2)
brief(img1, img2)
orb(img1, img2)

img1 = cv2.imread('D:\\NUST\\8th Semester\\Computer Vision\\Labs\\Assignment 1\\building_1.jpg')
img2 = cv2.imread('D:\\NUST\\8th Semester\\Computer Vision\\Labs\\Assignment 1\\building_2.jpg')
img3 = cv2.imread('D:\\NUST\\8th Semester\\Computer Vision\\Labs\\Assignment 1\\building_3.jpg')

sift(img1, img2)
surf(img1, img2)
brief(img1, img2)
orb(img1, img2)

sift(img1, img3)
surf(img1, img3)
brief(img1, img3)
orb(img1, img3)

sift(img2, img3)
surf(img2, img3)
brief(img2, img3)
orb(img2, img3)

