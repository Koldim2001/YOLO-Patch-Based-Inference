import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random
import numpy as np

from elements.CropElement import CropElement

class MakeCropsDetectThem:
    # Класс, реализующий нарезку на кропы и прогон через сеть
    def __init__(
        self,
        image: np.ndarray,
        model_path='yolov8m.pt',
        imgsz=640, conf=0.5, iou=0.7, 
        segment=False, 
        shape_x=700, shape_y=700,
        overlap_x=25, overlap_y=25, 
        show_crops=False,
        resize_results=False,
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
        self.show_crops = show_crops  # нужно ли делать визуализацию нарезки кропов
        self.resize_results = resize_results  # нужно ли делать приведение к исходным размерам изображения (медленная опреация)

        self.class_names_dict = self.model.names

        self.crops = self.get_crops_xy(self.image, shape_x=self.shape_x, shape_y=self.shape_y,
                 overlap_x=self.overlap_x, overlap_y=self.overlap_y, show=self.show_crops, return_crop_elements=True)
        self._detect_objects()



        
    def get_crops_xy(self, image_full, shape_x: int, shape_y: int,
                 overlap_x=15, overlap_y=15, show=False, return_crop_elements=False):
        """Preprocessing of the image. Generating crops with overlapping.

        Args:
            image_full (array): numpy array of an RGB image
            
            shape_x (int): size of the crop in the x-coordinate
            
            shape_y (int): size of the crop in the y-coordinate
            
            overlap_x (float, optional): Percentage of overlap along the x-axis
                    (how much subsequent crops borrow information from previous ones)

            overlap_y (float, optional): Percentage of overlap along the y-axis
                    (how much subsequent crops borrow information from previous ones)

            show (bool): enables the mode to display images using plt.imshow

        """    
        cross_koef_x = 1 - (overlap_x / 100)
        cross_koef_y = 1 - (overlap_y / 100)

        data_all_crops = []

        y_steps = int((image_full.shape[0] - shape_y) / (shape_y * cross_koef_y)) + 1
        x_steps = int((image_full.shape[1] - shape_x) / (shape_x * cross_koef_x)) + 1

        y_new = round((y_steps-1) * (shape_y * cross_koef_y) + shape_y)
        x_new = round((x_steps-1) * (shape_x * cross_koef_x) + shape_x)
        image_innitial = image_full.copy()
        image_full = cv2.resize(image_full, (x_new, y_new))

        if show:
            plt.figure(figsize=[x_steps*0.9, y_steps*0.9])

        count = 0
        for i in range(y_steps):
            for j in range(x_steps):
                x_start = int(shape_x * j * cross_koef_x)
                y_start = int(shape_y * i * cross_koef_y)

                # Check for residuals
                if x_start + shape_x > image_full.shape[1]:
                    print('Error in generating crops along the x-axis')
                    continue
                if y_start + shape_y > image_full.shape[0]:
                    print('Error in generating crops along the y-axis')
                    continue

                im_temp = image_full[y_start:y_start + shape_y, x_start:x_start + shape_x]

                # Display the result:
                if show:
                    plt.subplot(y_steps, x_steps, i * x_steps + j + 1)
                    plt.imshow(cv2.cvtColor(im_temp.copy(), cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                count += 1

                if return_crop_elements:
                    data_all_crops.append(CropElement(
                                            source_image=image_innitial,
                                            source_image_resized=image_full,
                                            crop=im_temp,
                                            number_of_crop=count,
                                            x_start=x_start,
                                            y_start=y_start,
                    ))
                else:
                    data_all_crops.append(im_temp)
        if show:
            plt.show()
            print('Number of generated images:', count)

        return data_all_crops


    def _detect_objects(self):
        for crop in self.crops:
            crop.calculate_inference(self.model, imgsz=self.imgsz, conf=self.conf, iou=self.iou, segment=self.segment)
            crop.calculate_real_values()
            if self.resize_results:
                crop.resize_results()