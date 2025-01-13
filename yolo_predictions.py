# YOLO Predictions
# Imports
import cv2 as cv
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        """
        Initializes the YOLO model.

        Args:
            onnx_model: path to the onnx weights.
            data_yaml: path to data_yaml file for YOLO model.

        Returns:
            None
        """
    
        # Load YAML
        with open("data.yaml", mode="r") as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml["names"]
        self.nc = data_yaml["nc"]

        # Load YOLO Model
        self.yolo = cv.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv.dnn.DNN_TARGET_CPU_FP16)

    def get_detection_data(self, row, x_factor, y_factor, confidences, boxes, classes):
        """
        Helper function for predictions().
        """
        CONFIDENCE_THRESHOLD = 0.4
        PROBABILITY_THRESHOLD = 0.25

        confidence = row[4] # Confidence is in the 5th column of the row
        if confidence > CONFIDENCE_THRESHOLD:
            class_score = row[5:].max() # Take the maximum probability of 10 objects possible
            class_id = row[5:].argmax() # Get the index position at which maximum probability occurs

            if class_score > PROBABILITY_THRESHOLD:
                cx, cy, w, h = row[:4]

                # Construct the bounding box from the four values
                # Get left, top, width, and height
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])

                # Append values into respective lists
                confidences.append(confidence)
                boxes.append(box)
                classes.append(class_id)

    def predictions(self, image):
        """
        Gets predictions using the YOLO model.

        Args:
            image: an image read by OpenCV; that is, the result of cv2.imread(image)

        Returns:
            An image containing the predictions from the YOLO model.
        """
        # Get number of rows, cols, and depth from image shape
        row, col, d = image.shape

        # Get the YOLO prediction from the image
        # First, convert image to square image
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Then, get predictions from square array
        INPUT_WH_YOLO = 640
        blob = cv.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() # Get predictions from YOLO Model

        # Non Maximum Supression
        # Step 1: Filter detections based on confidence score and probability score
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # Calculate the width and height of the input image
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/INPUT_WH_YOLO
        y_factor = image_h/INPUT_WH_YOLO

        # Approach 1
        # for i in range(len(detections)):
        #     row = detections[i]
        #     confidence = row[4] # Confidence is in the 5th column of the row
        #     if confidence > CONFIDENCE_THRESHOLD:
        #         class_score = row[5:].max() # Take the maximum probability of 10 objects possible
        #         class_id = row[5:].argmax() # Get the index position at which maximum probability occurs

        #         if class_score > PROBABILITY_THRESHOLD:
        #             cx, cy, w, h = row[:4]

        #             # Construct the bounding box from the four values
        #             # Get left, top, width, and height
        #             left = int((cx - 0.5 * w) * x_factor)
        #             top = int((cy - 0.5 * h) * y_factor)
        #             width = int(w * x_factor)
        #             height = int(h * y_factor)

        #             box = np.array([left, top, width, height])

        #             # Append values into respective lists
        #             confidences.append(confidence)
        #             boxes.append(box)
        #             classes.append(class_id)

        # Approach 2
        [self.get_detection_data(detections[i], x_factor, y_factor, confidences, boxes, classes) for i in range(len(detections))]

        # Cleaning
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # Try to apply non-maximum suppression
        try:
            indexes = cv.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45).flatten()
        except AttributeError:
            return image

        # Draw the Bounding Box
        for i in indexes:
            # Extract Bounding Boxes
            x, y, w, h = boxes_np[i]
            bb_conf = int(confidences_np[i] * 100)
            class_id = classes[i]
            class_name = self.labels[class_id]
            colors = self.generate_colors(class_id)

            text = f"{class_name}: {bb_conf}%"
            
            cv.rectangle(image, (x, y), (x + w, y + h), colors, 2)
            cv.rectangle(image, (x, y - 30), (x + w, y), colors, -1)
            cv.putText(image, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        return image
    
    def generate_colors(self, id):
        """
        Generates a unique color for each object class.

        Args:
            id: ID of a class.
        
        Returns:
            A tuple containing the RGB color values associated with an ID.
        """
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[id])
