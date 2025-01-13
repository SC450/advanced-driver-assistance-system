# Advanced Driver Assistance System
# Imports
import cv2 as cv
from yolo_predictions import YOLO_Pred
from lane_detection import LaneDetector

# Get video
vid_file = "dashcam_data/2.mp4"
vid = cv.VideoCapture(vid_file)

# Get YOLO model
yolo = YOLO_Pred("Model/weights/v4_best.onnx", "data.yaml")

# Get Lane Detector
lane_detector = LaneDetector()

# Play the video
i = 0
while True:
    # Read the frame
    ret, frame = vid.read()

    # Loop the video
    if not ret:
        vid.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    # Lane detection pipeline
    canny = lane_detector.make_canny(frame)
    masked = lane_detector.lane_mask(canny)
    lines = lane_detector.hough_lines(masked)
    hough_drawn = frame

    # Make sure there are lines that can be detected
    if lines is not None:
        # Draw hough lines (detected lane lines)
        hough_drawn = lane_detector.draw_lines(frame, lines)

        # Get YOLO predictions
        pred_frame = yolo.predictions(hough_drawn)

        # Show the frame
        cv.imshow("Advanced Driver Assistance System", pred_frame)
    else:
        pred_frame = yolo.predictions(hough_drawn)
        cv.imshow("Advanced Driver Assistance System", pred_frame)

    if cv.waitKey(25) == 27:
        break

    # For debugging
    print(i)
    i += 1

vid.release()
cv.destroyAllWindows()
