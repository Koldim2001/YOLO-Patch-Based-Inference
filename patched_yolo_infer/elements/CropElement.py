import numpy as np
import cv2


class CropElement:
    # Class containing information about a specific crop
    def __init__(
        self,
        source_image: np.ndarray,
        source_image_resized: np.ndarray,
        crop: np.ndarray,
        number_of_crop: int,
        x_start: int,
        y_start: int
    ) -> None:
        self.source_image = source_image  # Original image 
        self.source_image_resized = source_image_resized  # Original image (resized to a multiple of the crop size)
        self.crop = crop  # Specific crop 
        self.number_of_crop = number_of_crop  # Crop number in order from left to right, top to bottom
        self.x_start = x_start  # Coordinate of the top-left corner X
        self.y_start = y_start  # Coordinate of the top-left corner Y

        # YOLO output results:
        self.detected_conf = None  # List of confidence scores of detected objects
        self.detected_cls = None  # List of classes of detected objects
        self.detected_xyxy = None  # List of lists containing xyxy box coordinates
        self.detected_masks = None # List of np arrays containing masks in case of yolo-seg
        self.polygons = None # List of polygons points in case of using memory optimaze
        
        # Refined coordinates according to crop position information
        self.detected_xyxy_real = None  # List of lists containing xyxy box coordinates in values from source_image_resized or source_image
        self.detected_masks_real = None # List of np arrays containing masks in case of yolo-seg with the size of source_image_resized or source_image
        self.detected_polygons_real = None # List of polygons points in case of using memory optimaze in values from source_image_resized or source_image

    def calculate_inference(self, model, imgsz=640, conf=0.35, iou=0.7, segment=False, classes_list=None, memory_optimize=False, extra_args=None):

        # Perform inference
        extra_args = {} if extra_args is None else extra_args
        predictions = model.predict(self.crop, imgsz=imgsz, conf=conf, iou=iou, classes=classes_list, verbose=False, **extra_args)

        pred = predictions[0]

        # Get the bounding boxes and convert them to a list of lists
        self.detected_xyxy = pred.boxes.xyxy.cpu().int().tolist()

        # Get the classes and convert them to a list
        self.detected_cls = pred.boxes.cls.cpu().int().tolist()

        # Get the mask confidence scores
        self.detected_conf = pred.boxes.conf.cpu().numpy()

        if segment and len(self.detected_cls) != 0:
            if memory_optimize:
                # Get the polygons
                self.polygons = [mask.astype(np.uint16) for mask in pred.masks.xy]
            else:
                # Get the masks
                self.detected_masks = pred.masks.data.cpu().numpy().astype(np.uint8)
            

    def calculate_real_values(self):
        # Calculate real values of bboxes and masks in source_image_resized
        x_start_global = self.x_start  # Global X coordinate of the crop
        y_start_global = self.y_start  # Global Y coordinate of the crop

        self.detected_xyxy_real = []  # List of lists with xyxy box coordinates in the values ​​of the source_image_resized
        self.detected_masks_real = []  # List of np arrays with masks in case of yolo-seg sized as source_image_resized
        self.detected_polygons_real = [] # List of polygons in case of yolo-seg sized as source_image_resized

        for bbox in self.detected_xyxy:
            # Calculate real box coordinates based on the position information of the crop
            x_min, y_min, x_max, y_max = bbox
            x_min_real = x_min + x_start_global
            y_min_real = y_min + y_start_global
            x_max_real = x_max + x_start_global
            y_max_real = y_max + y_start_global
            self.detected_xyxy_real.append([x_min_real, y_min_real, x_max_real, y_max_real])

        if self.detected_masks is not None:
            for mask in self.detected_masks:
                # Create a black image with the same size as the source image
                black_image = np.zeros((self.source_image_resized.shape[0], self.source_image_resized.shape[1]))
                mask_resized = cv2.resize(np.array(mask).copy(), (self.crop.shape[1], self.crop.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                # Place the mask in the correct position on the black image
                black_image[y_start_global:y_start_global+self.crop.shape[0],
                            x_start_global:x_start_global+self.crop.shape[1]] = mask_resized

                # Append the masked image to the list of detected_masks_real
                self.detected_masks_real.append(black_image.astype(np.uint8))

        if self.polygons is not None:
            # Adjust the mask coordinates
            for mask in self.polygons:
                mask[:, 0] += x_start_global  # Add x_start_global to all x coordinates
                mask[:, 1] += y_start_global  # Add y_start_global to all y coordinates
                self.detected_polygons_real.append(mask.astype(np.uint16))
        
    def resize_results(self):
        # from source_image_resized to source_image sizes transformation
        resized_xyxy = []
        resized_masks = []
        resized_polygons = []

        for bbox in self.detected_xyxy_real:
            # Resize bbox coordinates
            x_min, y_min, x_max, y_max = bbox
            x_min_resized = int(x_min * (self.source_image.shape[1] / self.source_image_resized.shape[1]))
            y_min_resized = int(y_min * (self.source_image.shape[0] / self.source_image_resized.shape[0]))
            x_max_resized = int(x_max * (self.source_image.shape[1] / self.source_image_resized.shape[1]))
            y_max_resized = int(y_max * (self.source_image.shape[0] / self.source_image_resized.shape[0]))
            resized_xyxy.append([x_min_resized, y_min_resized, x_max_resized, y_max_resized])

        for mask in self.detected_masks_real:
            # Resize mask
            mask_resized = cv2.resize(mask, (self.source_image.shape[1], self.source_image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            resized_masks.append(mask_resized.astype(np.uint8))


        for polygon in self.detected_polygons_real:
            polygon[:, 0] = (polygon[:, 0] * (self.source_image.shape[1] / self.source_image_resized.shape[1])).astype(np.uint16)
            polygon[:, 1] = (polygon[:, 1] * (self.source_image.shape[0] / self.source_image_resized.shape[0])).astype(np.uint16)
            resized_polygons.append(polygon)

        self.detected_xyxy_real = resized_xyxy
        self.detected_masks_real = resized_masks
        self.detected_polygons_real = resized_polygons
