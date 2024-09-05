import cv2
import numpy as np
import os
import glob

checkerBoard = (6,9)
# termination criteria: stops the algorithm if specified accuracy (epsilon) OR specified number of iterations (max_iter) is reached
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
