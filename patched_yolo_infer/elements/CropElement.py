import numpy as np
import cv2


class CropElement:
    # Класс, содержаций информацию о конкретном кропе
    def __init__(
        self,
        source_image: np.ndarray,
        source_image_resized: np.ndarray,
        crop: np.ndarray,
        number_of_crop: int,
        x_start: int,
        y_start: int
    ) -> None:
        self.source_image = source_image  # Исходное изображение 
        self.source_image_resized = source_image_resized  # Исходное изображение (ресайзнутое до кратного размера кропам)
        self.crop = crop  # Конкретный кроп 
        self.number_of_crop = number_of_crop  # Номер кропа по порядку слева направо сверху вниз
        self.x_start = x_start  # Координата верхнего левого угла Х
        self.y_start = y_start  # Координата верхнего левого угла Y
        # Результаты на выходе с YOLO:
        self.detected_conf = None  # Список уверенностей задетектированных объектов
        self.detected_cls = None  # Список классов задетектированных объектов
        self.detected_xyxy = None  # Список списков с координатами xyxy боксов
        self.detected_masks = None # Список np массивов с масками в случае yolo-seg
        
        # Уточненные координаты согласноинформации о полодении кропа
        self.detected_xyxy_real = None  # Список списков с координатами xyxy боксов в значениях от source_image_resized
        self.detected_masks_real = None # Список np массивов с масками в случае yolo-seg размером как source_image_resized

    def calculate_inference(self, model, imgsz=640, conf=0.35, iou=0.7, segment=False, classes_list=None):
        # Perform inference

        predictions = model.predict(self.crop, imgsz=imgsz, conf=conf, iou=iou, classes=classes_list, verbose=False)

        pred = predictions[0]

        # Get the bounding boxes and convert them to a list of lists
        self.detected_xyxy = pred.boxes.xyxy.cpu().int().tolist()

        # Get the classes and convert them to a list
        self.detected_cls = pred.boxes.cls.cpu().int().tolist()

        # Get the mask confidence scores
        self.detected_conf = pred.boxes.conf.cpu().numpy()

        if segment and len(self.detected_cls) != 0:
            # Get the masks
            self.detected_masks = pred.masks.data.cpu().numpy()

    def calculate_real_values(self):
        # Calculate real values of bboxes and masks
        x_start_global = self.x_start  # Global X coordinate of the crop
        y_start_global = self.y_start  # Global Y coordinate of the crop

        self.detected_xyxy_real = []  # List of lists with xyxy box coordinates in the values ​​of the source_image_resized
        self.detected_masks_real = []  # List of np arrays with masks in case of yolo-seg sized as source_image_resized

        for bbox in self.detected_xyxy:
            # Calculate real box coordinates based on the position information of the crop
            x_min, y_min, x_max, y_max = bbox
            x_min_real = x_min + x_start_global
            y_min_real = y_min + y_start_global
            x_max_real = x_max + x_start_global
            y_max_real = y_max + y_start_global
            self.detected_xyxy_real.append([x_min_real, y_min_real, x_max_real, y_max_real])

        if self.detected_masks is not None:
            self.detected_masks_real = []
            for mask in self.detected_masks:
                # Create a black image with the same size as the source image
                black_image = np.zeros((self.source_image_resized.shape[0], self.source_image_resized.shape[1]))
                mask_resized = cv2.resize(np.array(mask).copy(), (self.crop.shape[1], self.crop.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                # Place the mask in the correct position on the black image
                black_image[y_start_global:y_start_global+self.crop.shape[0],
                            x_start_global:x_start_global+self.crop.shape[1]] = mask_resized

                # Append the masked image to the list of detected_masks_real
                self.detected_masks_real.append(black_image)

    def resize_results(self):
        resized_xyxy = []
        resized_masks = []

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
            resized_masks.append(mask_resized)

        self.detected_xyxy_real = resized_xyxy
        self.detected_masks_real = resized_masks
