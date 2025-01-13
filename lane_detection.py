# Imports
import cv2 as cv
import numpy as np

class LaneDetector:
    def make_canny(self, img):
        """
        Makes an image canny.

        Args:
            img: Image to be converted to Canny.

        Returns:
            A Canny version of the img.
        """

        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        KERNEL_SIZE = 3
        blur = cv.GaussianBlur(gray, (KERNEL_SIZE, KERNEL_SIZE), cv.BORDER_DEFAULT)

        # Apply Canny Edge Detection
        lower_thres = 50
        higher_thres = 150
        canny = cv.Canny(blur, lower_thres, higher_thres)

        # Return the canny image
        return canny

    def lane_mask(self, img):
        """
        Returns a lower trapezoidal shaped region of the input image.

        Args:
            img: Canny version of original image.

        Returns:
            The canny image with only a lower region shown.
        """

        # Create a mask image with the same size as the image
        mask = np.zeros_like(img)

        # Create the region of interest in our mask image; define vertices first (trapezoid at bottom)
        rows, cols = img.shape[:2]
        bottom_left_vert = [cols * 0.1, rows * 0.95]
        top_left_vert = [cols * 0.4, rows * 0.65]
        bottom_right_vert = [cols * 0.9, rows * 0.95]
        top_right_vert = [cols * 0.6, rows * 0.65]
        verts = np.array([[bottom_left_vert, top_left_vert, top_right_vert, bottom_right_vert]], dtype=np.int32)

        # Create the trapezoid
        white = (255, 255, 255)
        cv.fillPoly(mask, verts, white)

        # Return the masked image
        #cv.imshow("Mask", mask) # for debugging
        return cv.bitwise_and(img, mask)

    def lane_mask_color(self, img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        low_white = np.array([68, 82, 88])
        high_white = np.array([253, 253, 253])
        mask = cv.inRange(hsv, low_white, high_white)
        return cv.Canny(mask, 75, 150)

    def hough_lines(self, img):
        """
        Gets the Hough lines of an image.

        Args:
            img: Masked version of the original image

        Returns
            An array of the Hough lines.
        """
        lines = cv.HoughLinesP(img, 1, np.pi/180, threshold=60, maxLineGap=900, minLineLength=100)
        return lines

    def draw_lines(self, img, lines):
        """
        Draws Hough lines on the original image.

        Args:
            img: The original image.
            lines: Array of hough lines; result from hough_lines().

        Returns:
            Image with hough lines drawn over it.
        """
        [cv.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 3) for line in lines]
        return img
