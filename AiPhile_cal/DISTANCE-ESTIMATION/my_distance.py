import cv2 as cv
from cv2 import aruco
import numpy as np

calib_data_path = "../calib_data/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 6.3  # centimeters

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Convert the grayscale frame to low saturation
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_frame[:, :, 1] = hsv_frame[:, :, 1] * 0.3  # Adjust the saturation factor
    low_saturation_frame = cv.cvtColor(hsv_frame, cv.COLOR_HSV2BGR)

    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        low_saturation_frame, marker_dict, parameters=param_markers
    )
    
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef
        )
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(
                low_saturation_frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            # ... (rest of your code remains unchanged)
            
    cv.imshow("frame", low_saturation_frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
