import numpy as np

"""Returns the bounding box for the right hand along with a boolean indicating if the right hand was even found"""
def rightHandBox(detectionResults, visibilityThreshold, height, width):
  poseLandmarkers = detectionResults.pose_landmarks[0]
  rWristHolder = poseLandmarkers[16]
  rPinkyHolder = poseLandmarkers[18]
  rIndexHolder = poseLandmarkers[20]
  rThumbHolder = poseLandmarkers[22]

  cordsRWrist = np.array([rWristHolder.x*width, rWristHolder.y*height])
  cordsRIndex = np.array([rIndexHolder.x*width, rIndexHolder.y*height])

#If visibility below threshold return 0,0,0,0 as bounding box and false as found hand
  if rWristHolder.visibility < visibilityThreshold and rIndexHolder.visibility < visibilityThreshold and rPinkyHolder.visibility < visibilityThreshold and rThumbHolder.visibility < visibilityThreshold:

    return [0,0,0,0], False

  centerXYR, directionVector, projectedWrist = getCircleUnitV(cordsRIndex, cordsRWrist)
  cornerCords = getBoxCutoffAtWrist(centerXYR, projectedWrist, directionVector)

  return cornerCords, True

def leftHandBox(detectionResults, visibilityThreshold, height, width):
  poseLandmarkers = detectionResults.pose_landmarks[0]
  lWristHolder = poseLandmarkers[15]
  lPinkyHolder = poseLandmarkers[17]
  lIndexHolder = poseLandmarkers[19]
  lThumbHolder = poseLandmarkers[21]

  cordsLWrist = np.array([lWristHolder.x*width, lWristHolder.y*height])
  cordsLIndex = np.array([lIndexHolder.x*width, lIndexHolder.y*height])

  if lWristHolder.visibility < visibilityThreshold and lIndexHolder.visibility < visibilityThreshold and lPinkyHolder.visibility < visibilityThreshold and lThumbHolder.visibility < visibilityThreshold:
    return [0,0,0,0], False

  centerXYR, directionVector, projectedWrist = getCircleUnitV(cordsLIndex, cordsLWrist)
  cornerCords = getBoxCutoffAtWrist(centerXYR, projectedWrist, directionVector)

  return cornerCords, True


"""Returns a center xy radius coordinate for the bounding box just calculates the distance between the index knuckle and the wrist then mulitplies it by some delta and makes it the radius.
Can think of the created bounding box as a center XYWH where the Width and Height are 2 radius"""
def getCircleUnitV(index, wrist, delta=2.25):

  indexVector = np.array(index-wrist)
  indexMagnitude = np.linalg.norm(indexVector)
  centerX = int(index[0])
  centerY = int(index[1])
  radius = int(indexMagnitude * delta)
  projectedWrist = wrist - indexVector/3
  directionVector = np.array([indexVector[0]/indexMagnitude, indexVector[1]/indexMagnitude])

  return [centerX, centerY, radius], directionVector, projectedWrist.astype(int)

"""Outputs the Top Left Bottom Right coordinates of a box that should not include too much beyond the wrist. Estimates orientation of the hand from the direction vector (wrist to index unit vector)
If hand is oriented horizontally then the horizontal portion of the direction vector should be greater than 0.7 (abitrary value chosen) and we should cut off the x part of the box at the wrist."""
def getBoxCutoffAtWrist(centerXYR, projectedWrist, directionVector):
  tLBRCords = [centerXYR[0]-centerXYR[2], centerXYR[1]-centerXYR[2], centerXYR[0]+centerXYR[2], centerXYR[1]+centerXYR[2]]
  
  #If horizontal(x) component greater than 0.7 adjust x part of the box
  if abs(directionVector[0]) > 0.7:
    if directionVector[0] < 0:
      tLBRCords[2] = projectedWrist[0]#direction of x is negative means hand pointed towards left so adjust right side of box.
    else:
      tLBRCords[0] = projectedWrist[0]
  
  elif abs(directionVector[1]) > 0.7:
    if directionVector[1] < 0:
      tLBRCords[3] = projectedWrist[1]
    else:
      tLBRCords[1] = projectedWrist[1]

  return tLBRCords
  



    


  
  
  