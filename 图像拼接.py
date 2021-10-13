import numpy as np
import cv2


class Stitcher():
    def __init__(self):
        pass

    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatchers=False):
        imageB, imageA = images
        kpsA, featuresA = self.detectAndDescribe(imageA)
        kpsB, featuresB = self.detectAndDescribe(imageB)
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        if M is None:
            return None
        matches, H, status = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        if showMatchers:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return result, vis
        return result
