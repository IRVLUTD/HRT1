import mediapipe as mp
import HumanPoseHandBoxes.MPLandmarking as MPL
import numpy as np
import cv2
import os
import HumanPoseHandBoxes.CalculateBoxes as CB

"""Given 3 images attempts to make them one image with the original on the left, the right hand bounding box on the top right, and the left hand bounding box on the bottom right
Scales up the images if needed more for visual purposes"""
def concatenate_images_with_padding(imageWithBB, leftHandImage, rightHandImage):
    imageHeight = imageWithBB.shape[0]

    def resize_image(image, target_height):
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return image
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_LINEAR)

    leftHandImage_resized = resize_image(leftHandImage, imageHeight // 2)
    rightHandImage_resized = resize_image(rightHandImage, imageHeight // 2)

    left_width = leftHandImage_resized.shape[1]
    right_width = rightHandImage_resized.shape[1]

    max_width = max(left_width, right_width)

    left_padded = np.zeros((leftHandImage_resized.shape[0], max_width, 3), dtype=np.uint8)
    right_padded = np.zeros((rightHandImage_resized.shape[0], max_width, 3), dtype=np.uint8)

    left_padded[:, :left_width] = leftHandImage_resized
    right_padded[:, :right_width] = rightHandImage_resized

    stacked_images = np.vstack((left_padded, right_padded))

    total_height = stacked_images.shape[0]
    height_diff = imageHeight - total_height
    if height_diff > 0:
        padding = np.zeros((height_diff, stacked_images.shape[1], 3), dtype=np.uint8)
        stacked_images = np.vstack((stacked_images, padding))

    final_image = np.hstack((imageWithBB, stacked_images))

    return final_image

"""Given an image and the top left bottom right coordinates draws the box on the image in that color"""
def drawBoxGivenCorners(image, cornerCords, color):
  topLeft = (cornerCords[0], cornerCords[1])
  bottomRight = (cornerCords[2], cornerCords[3])

  cv2.rectangle(image, topLeft, bottomRight, color, 1)
  return topLeft, bottomRight

"""Gets the bounding boxes around the 2 hands in an image, returns original image with the bounding boxes, 2 separate images for the left and right bounding boxes, and the topleft bottomright coordinates of the boxes"""
def getImageWithBB(image, detectionResults, visibilityThreshold):
    imageCopy = np.copy(image)
    height = imageCopy.shape[0]
    width = imageCopy.shape[1]
    rTopLeft = [0,0]
    rBottomRight = [0,0]
    lTopLeft = [0,0]
    lBottomRight = [0,0]

    rCornerCords, rFoundHand = CB.rightHandBox(detectionResults, visibilityThreshold, height, width)
    if rFoundHand:
      rTopLeft, rBottomRight = drawBoxGivenCorners(imageCopy, rCornerCords, (0,255,0))

    lCornerCords, lFoundHand = CB.leftHandBox(detectionResults, visibilityThreshold,height, width)
    if lFoundHand:
      lTopLeft, lBottomRight = drawBoxGivenCorners(imageCopy, lCornerCords, (255,0,0))

    rightHandImg = image[rTopLeft[1]:rBottomRight[1], rTopLeft[0]:rBottomRight[0]]
    rBB = np.array([rTopLeft[0], rTopLeft[1], rBottomRight[0], rBottomRight[1]])

    leftHandImg = image[lTopLeft[1]:lBottomRight[1], lTopLeft[0]:lBottomRight[0]]
    lBB = np.array([lTopLeft[0], lTopLeft[1], lBottomRight[0], lBottomRight[1]])

    imageWithBoxes = imageCopy

    return imageWithBoxes, leftHandImg, rightHandImg, rBB, lBB

"""Given an image if either the height or width is below the threshold the image gets scaled up. More for visual purposes."""
def scaleImageToThreshold(image, threshold):
  height, width = image.shape[:2]
  if height == 0 or width == 0:
    return image

  if height < threshold and width < threshold:
    scaleFactor = threshold / min(height,width)
    newHeight = int(height * scaleFactor)
    newWidth = int(width * scaleFactor)
    resizedImage = cv2.resize(image, (newWidth, newHeight))
    return resizedImage
  else:
    return image
  