import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from ..elements.CropElement import CropElement


class MakeCropsDetectThem:
    """
    Class implementing cropping and passing crops through a neural network
    for detection/segmentation.

    Args:
        image (np.ndarray): Input image BGR.
        model_path (str): Path to the YOLO model.
        imgsz (int): Size of the input image for inference YOLO.
        conf (float): Confidence threshold for detections YOLO.
        iou (float): IoU threshold for non-maximum suppression YOLOv8 of single crop.
        classes_list (List[int] or None): List of classes to filter detections. If None, 
                                          all classes are considered. Defaults to None.
        segment (bool): Whether to perform segmentation (YOLO-seg).
        shape_x (int): Size of the crop in the x-coordinate.
        shape_y (int): Size of the crop in the y-coordinate.
        overlap_x (int): Percentage of overlap along the x-axis.
        overlap_y (int): Percentage of overlap along the y-axis.
        show_crops (bool): Whether to visualize the cropping.
        resize_initial_size (bool): Whether to resize the results to the original 
                                    image size (ps: slow operation).
        model: Pre-initialized model object. If provided, the model will be used directly 
                   instead of loading from model_path.
        memory_optimize (bool): Memory optimization option for segmentation (less accurate results)
        batch_inference (bool): Batch inference of image crops through a neural network instead of 
                    sequential passes of crops (ps: Faster inference, higher memory use)
        inference_extra_args (dict): Dictionary with extra ultralytics inference parameters

    Attributes:
        model: YOLOv8 model loaded from the specified path.
        image (np.ndarray): Input image BGR.
        imgsz (int): Size of the input image for inference.
        conf (float): Confidence threshold for detections.
        iou (float): IoU threshold for non-maximum suppression.
        classes_list (List[int] or None): List of classes to filter detections. If None, 
                                          all classes are considered. Defaults to None.
        segment (bool): Whether to perform segmentation (YOLO-seg).
        shape_x (int): Size of the crop in the x-coordinate.
        shape_y (int): Size of the crop in the y-coordinate.
        overlap_x (int): Percentage of overlap along the x-axis.
        overlap_y (int): Percentage of overlap along the y-axis.
        crops (list): List to store the CropElement objects.
        show_crops (bool): Whether to visualize the cropping.
        resize_initial_size (bool): Whether to resize the results to the original  
                                    image size (ps: slow operation).
        class_names_dict (dict): Dictionary containing class names of the YOLO model.
        memory_optimize (bool): Memory optimization option for segmentation (less accurate results)
        batch_inference (bool): Batch inference of image crops through a neural network instead of 
                                    sequential passes of crops (ps: Faster inference, higher memory use)
        inference_extra_args (dict): Dictionary with extra ultralytics inference parameters
    """
    def __init__(
        self,
        image: np.ndarray,
        model_path="yolov8m.pt",
        imgsz=640,
        conf=0.5,
        iou=0.7,
        classes_list=None,
        segment=False,
        shape_x=700,
        shape_y=600,
        overlap_x=25,
        overlap_y=25,
        show_crops=False,
        resize_initial_size=True,
        model=None,
        memory_optimize=True,
        inference_extra_args=None,
        batch_inference=False,
    ) -> None:
        if model is None:
            self.model = YOLO(model_path)  # Load the model from the specified path
        else:
            self.model = model
        self.image = image  # Input image
        self.imgsz = imgsz  # Size of the input image for inference
        self.conf = conf  # Confidence threshold for detections
        self.iou = iou  # IoU threshold for non-maximum suppression
        self.classes_list = classes_list  # Classes to detect
        self.segment = segment  # Whether to perform segmentation
        self.shape_x = shape_x  # Size of the crop in the x-coordinate
        self.shape_y = shape_y  # Size of the crop in the y-coordinate
        self.overlap_x = overlap_x  # Percentage of overlap along the x-axis
        self.overlap_y = overlap_y  # Percentage of overlap along the y-axis
        self.crops = []  # List to store the CropElement objects
        self.show_crops = show_crops  # Whether to visualize the cropping
        self.resize_initial_size = resize_initial_size  # slow operation !
        self.memory_optimize = memory_optimize # memory opimization option for segmentation
        self.class_names_dict = self.model.names # dict with human-readable class names
        self.inference_extra_args = inference_extra_args # dict with extra ultralytics inference parameters
        self.batch_inference = batch_inference # batch inference of image crops through a neural network

        self.crops = self.get_crops_xy(
            self.image,
            shape_x=self.shape_x,
            shape_y=self.shape_y,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            show=self.show_crops,
        )
        if self.batch_inference:
            self._detect_objects_batch() 
        else:
            self._detect_objects()

    def get_crops_xy(
        self,
        image_full,
        shape_x: int,
        shape_y: int,
        overlap_x=25,
        overlap_y=25,
        show=False,
    ):
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
        batch_of_crops = []

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

                data_all_crops.append(CropElement(
                                        source_image=image_innitial,
                                        source_image_resized=image_full,
                                        crop=im_temp,
                                        number_of_crop=count,
                                        x_start=x_start,
                                        y_start=y_start,
                ))
                if self.batch_inference:
                    batch_of_crops.append(im_temp)

        if show:
            plt.show()
            print('Number of generated images:', count)

        if self.batch_inference:
            return data_all_crops, batch_of_crops
        else:
            return data_all_crops

    def _detect_objects(self):
        """
        Method to detect objects in each crop.

        This method iterates through each crop, performs inference using the YOLO model,
        calculates real values, and optionally resizes the results.

        Returns:
            None
        """
        for crop in self.crops:
            crop.calculate_inference(
                self.model,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                segment=self.segment,
                classes_list=self.classes_list,
                memory_optimize=self.memory_optimize,
                extra_args=self.inference_extra_args
            )
            crop.calculate_real_values()
            if self.resize_initial_size:
                crop.resize_results()

    def _detect_objects_batch(self):
        """
        Method to detect objects in batch of image crops.

        This method performs batch inference using the YOLO model,
        calculates real values, and optionally resizes the results.

        Returns:
            None
        """
        crops, batch = self.crops
        self.crops = crops
        self._calculate_batch_inference(
            batch,
            self.crops,
            self.model,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            segment=self.segment,
            classes_list=self.classes_list,
            memory_optimize=self.memory_optimize,
            extra_args=self.inference_extra_args
        )
        for crop in self.crops:
            crop.calculate_real_values()
            if self.resize_initial_size:
                crop.resize_results()

    def _calculate_batch_inference(
        self,
        batch,
        crops,
        model,
        imgsz=640,
        conf=0.35,
        iou=0.7,
        segment=False,
        classes_list=None,
        memory_optimize=False,
        extra_args=None,
    ):
        # Perform batch inference of image crops through a neural network
        extra_args = {} if extra_args is None else extra_args
        predictions = model.predict(
            batch,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=classes_list,
            verbose=False,
            **extra_args
        )

        for pred, crop in zip(predictions, crops):

            # Get the bounding boxes and convert them to a list of lists
            crop.detected_xyxy = pred.boxes.xyxy.cpu().int().tolist()

            # Get the classes and convert them to a list
            crop.detected_cls = pred.boxes.cls.cpu().int().tolist()

            # Get the mask confidence scores
            crop.detected_conf = pred.boxes.conf.cpu().numpy()

            if segment and len(crop.detected_cls) != 0:
                if memory_optimize:
                    # Get the polygons
                    crop.polygons = [mask.astype(np.uint16) for mask in pred.masks.xy]
                else:
                    # Get the masks
                    crop.detected_masks = pred.masks.data.cpu().numpy()
