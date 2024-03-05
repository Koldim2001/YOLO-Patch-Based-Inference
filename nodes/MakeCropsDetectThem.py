import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
import numpy as np

from elements.CropElement import CropElement

class MakeCropsDetectThem:
    # Класс, реализующий нарезку на кропы и прогон через сеть
    def __init__(
        model_path,
        image: np.ndarray,
        imgsz=640, conf=0.5, iou=0.7, 
        segment=False, 
        shape_x=700, shape_y=700,
        overlap_x=25, overlap_y=25,
    ) -> None:
        
        self.model = YOLO(model_path)  # Load the model from the specified path
        self.image = image  # Input image
        self.imgsz = imgsz  # Size of the input image for inference
        self.conf = conf  # Confidence threshold for detections
        self.iou = iou  # IoU threshold for non-maximum suppression
        self.segment = segment  # Whether to perform segmentation
        self.shape_x = shape_x  # Size of the crop in the x-coordinate
        self.shape_y = shape_y  # Size of the crop in the y-coordinate
        self.overlap_x = overlap_x  # Percentage of overlap along the x-axis
        self.overlap_y = overlap_y  # Percentage of overlap along the y-axis
        self.crops = []  # List to store the CropElement objects

        
    