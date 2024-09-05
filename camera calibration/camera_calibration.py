import cv2
import numpy as np
import os
import glob

# defining the number of rows and columns of INNER squares of checkerboard
checkerBoard = (6, 9)

# termination criteria: stops the algorithm if specified accuracy (epsilon) OR specified number of iterations (max_iter) is reached
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3d object points in world space from ALL images
all_obj_pts = []

# 2d image points in image plane from ALL images
all_img_pts = []

# initialise 3d object points for EACH image
obj_pts = np.zeros((1,checkerBoard[0] * checkerBoard[1], 3), np.float32) # creates a 3d array of zeros that are 32 bit floats (less accurate than 64 bits but saves memory)
obj_pts[0, :, :2] = np.mgrid[0:checkerBoard[0], 0:checkerBoard[1]].T.reshape(-1, 2) # reshapes into array with 2 columns, T is the same as np.transpose, m.grid does uhhhh something confusing

prev_image_shape = None

images = glob.glob('./images/*.jpg')
for fileName in images:
    image = cv2.imread(fileName)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find checkerboard corners
    result, corners = cv2.findChessboardCorners(gray_image, checkerBoard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) # result returns true when the adequate number of inner corners are found, flags are intended to help with checkerboard recognition

    if result == True:
        all_obj_pts.append(obj_pts)
        refined_corners = cv2.cornerSubPix(gray_image, (11,11), (-1,-1), criteria)
        all_img_pts.append(refined_corners)
        
        #draw and display
        image = cv2.drawChessboardCorners(image, checkerBoard, refined_corners, result)
        cv2.imshow('img',image)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

    else:
        print(f'Insufficient corners found for {fileName}.jpg!')
        cv2.waitKey(500)

result, intrinsic_parameters, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(all_obj_pts, all_img_pts, gray_image.shape[::-1], None, None)

print('----- Camera matrix -----')
print('Intrinsic parameters:')
print(intrinsic_parameters)
print()
print('Distortion coefficients:')
print(distortion_coefficients)
print()
print('Rotation vectors:')
print(rotation_vectors)
print()
print('Translation vectors:')
print(translation_vectors)