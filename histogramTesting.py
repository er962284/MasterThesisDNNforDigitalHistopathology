from skimage import exposure
import matplotlib.pyplot as plt
import argparse
import cv2

# gray_img = cv2.imread('x.jpg')
# lab_image = cv2.cvtColor(gray_img, cv2.COLOR_BGR2LAB)
# l_channel,a_channel,b_channel = cv2.split(lab_image)
# #print(l_channel)
# print(a_channel)
# # print(b_channel)
# #cv2.imshow('GoldenGate',gray_img)
#
# hist = cv2.calcHist([lab_image],[2],None,[256],[-128,127])
# plt.hist(gray_img.ravel(),256,[-128,127])
# plt.title('Histogram for gray scale picture')
# plt.show()

src = cv2.imread("x.jpg")
ref = cv2.imread("Unbent.jpg")
# determine if we are performing multichannel histogram matching
# and then perform histogram matching itself
print(" performing histogram matching...")
multi = True if src.shape[-1] > 1 else False
matched = exposure.match_histograms(src, ref, multichannel=multi)
# show the output images
cv2.imshow("Source", src)
cv2.imshow("Reference", ref)
cv2.imshow("Matched", matched)
cv2.waitKey(0)
