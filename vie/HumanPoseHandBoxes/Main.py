import cv2
import MPLandmarking as MPL
import OutputImages as OUT
import os


def main():
  imageFilePath = "Images\image.jpg"
  landmarks = MPL.mediaPipeOnImageFilePath(imageFilePath) #Get detection results
  image = cv2.imread(imageFilePath)

  imageWithBB, leftHandImage, rightHandImage, rBB, lBB = OUT.getImageWithBB(image, landmarks, .7)

  final_image = OUT.concatenate_images_with_padding(imageWithBB, leftHandImage, rightHandImage)

  cv2.imshow("FinalImage",final_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  filename = os.path.basename(imageFilePath)
  name, ext = os.path.splitext(filename)
  outputFolderPath = "OutputImages"

  outputFilePath = os.path.join(outputFolderPath, name + "_boxed" + ext)

  cv2.imwrite(outputFilePath, final_image)

if __name__ == "__main__":
  main()