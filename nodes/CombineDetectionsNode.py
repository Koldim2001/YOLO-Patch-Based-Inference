import cv2
import numpy as np
import torch
from elements.CropElement import CropElement
from nodes.MakeCropsDetectThem import MakeCropsDetectThem

class CombineDetectionsNode:
    # Класс, реализующий нарезку на кропы и прогон через сеть
    def __init__(
        self,
        element_crops: MakeCropsDetectThem,
        nms_iou=0.6,
        match_metric = 'IOU'
    ) -> None:
        self.conf_treshold = element_crops.conf
        self.class_names = element_crops.class_names_dict 
        self.crops = element_crops.crops  # List to store the CropElement objects
        if element_crops.resize_results:
            self.image = element_crops.crops[0].source_image
        else:
            self.image = element_crops.crops[0].source_image_resized
       
        self.nms_iou = nms_iou  # IoU treshold for NMS
        self.match_metric = match_metric
        (
            self.detected_conf_list_full,
            self.detected_xyxy_list_full,
            self.detected_masks_list_full,
            self.detected_cls_id_list_full
        ) = self.combinate_detections(crops=self.crops)

        self.detected_cls_names_list_full = [self.class_names[value] for value in self.detected_cls_id_list_full] # make str list

        # Вызываем метод nms для фильтрации предсказаний
        self.filtered_indices = self.nms(
            self.detected_conf_list_full,
            self.detected_xyxy_list_full,
            self.match_metric, 
            self.nms_iou
        )

        # Применяем фильтрацию к спискам предсказаний
        self.filtered_confidences = [self.detected_conf_list_full[i] for i in self.filtered_indices]
        self.filtered_boxes = [self.detected_xyxy_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_id = [self.detected_cls_id_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_names = [self.detected_cls_names_list_full[i] for i in self.filtered_indices]

        if element_crops.segment:
            self.filtered_masks = [self.detected_masks_list_full[i] for i in self.filtered_indices]
        else:
            self.filtered_masks = []

        
    def combinate_detections(self, crops):
        detected_conf = []
        detected_xyxy = []
        detected_masks = []
        detected_cls = []

        for crop in crops:
            detected_conf.extend(crop.detected_conf)
            detected_xyxy.extend(crop.detected_xyxy_real)
            detected_masks.extend(crop.detected_masks_real)
            detected_cls.extend(crop.detected_cls)

        return detected_conf, detected_xyxy, detected_masks, detected_cls

    '''
    def nms(self, confidences, boxes, iou_threshold):
        # Преобразуем формат bbox'ов из xyxy в x, y, width, height
        bboxes = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]

        # Вызываем функцию NMSBoxes
        picked_indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.conf_treshold, iou_threshold)

        # Выдаем индексы отфильтрованных предсказаний
        return picked_indices
    '''

    def nms(self, 
        confidences: list, 
        boxes: list, 
        match_metric,
        iou_threshold,
    ):
        """
        Apply non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            predictions: (tensor) The location preds for the image
                along with the class predscores, Shape: [num_boxes,5].
            match_metric: (str) IOU or IOS
            match_threshold: (float) The overlap thresh for
                match metric.
        Returns:
            A list of filtered indexes, Shape: [ ,]
        """

        # Convert lists to tensors
        boxes = torch.tensor(boxes)
        confidences = torch.tensor(confidences)

        # Extract coordinates for every prediction box present in P
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate area of every box in P
        areas = (x2 - x1) * (y2 - y1)

        # Sort the prediction boxes in P according to their confidence scores
        order = confidences.argsort()

        # Initialise an empty list for filtered prediction boxes
        keep = []

        while len(order) > 0:
            # Extract the index of the prediction with highest score
            # we call this prediction S
            idx = order[-1]

            # Push S in filtered predictions list
            keep.append(idx.tolist())

            # Remove S from P
            order = order[:-1]

            # Sanity check
            if len(order) == 0:
                break

            # Select coordinates of BBoxes according to the indices in order
            xx1 = torch.index_select(x1, dim=0, index=order)
            xx2 = torch.index_select(x2, dim=0, index=order)
            yy1 = torch.index_select(y1, dim=0, index=order)
            yy2 = torch.index_select(y2, dim=0, index=order)

            # Find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])

            # Find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1

            # Take max with 0.0 to avoid negative w and h
            # due to non-overlapping boxes
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            # Find the intersection area
            inter = w * h

            # Find the areas of BBoxes according the indices in order
            rem_areas = torch.index_select(areas, dim=0, index=order)

            # Calculate the distance between centers of the boxes
            # Centers of boxes: (x1 + x2) / 2, (y1 + y2) / 2
            cx = (x1[idx] + x2[idx]) / 2
            cy = (y1[idx] + y2[idx]) / 2
            rem_cx = (x1[order] + x2[order]) / 2
            rem_cy = (y1[order] + y2[order]) / 2
            dist_centers = ((cx - rem_cx) ** 2 + (cy - rem_cy) ** 2).sqrt()

            if match_metric == "IOU":
                # Find the union of every prediction T in P
                # with the prediction S
                # Note that areas[idx] represents area of S
                union = (rem_areas - inter) + areas[idx]
                # Find the IoU of every prediction in P with S
                match_metric_value = inter / union

            elif match_metric == "IOS":
                # Find the smaller area of every prediction T in P
                # with the prediction S
                # Note that areas[idx] represents area of S
                smaller = torch.min(rem_areas, areas[idx])
                # Find the IoU of every prediction in P with S
                match_metric_value = inter / smaller

            elif match_metric == "DIoU":
                # Calculate the diagonal distance between the boxes
                diag_dist = ((x2[idx] - x1[idx]) ** 2 + (y2[idx] - y1[idx]) ** 2).sqrt()
                # Calculate the IoU
                union = (rem_areas - inter) + areas[idx]
                iou = inter / union
                # Calculate DIoU
                match_metric_value = iou - dist_centers / (diag_dist + 1e-7)

            else:
                raise ValueError("Unknown matching metric")

            # Keep the boxes with IoU less than thresh_iou
            mask = match_metric_value < iou_threshold
            order = order[mask]

        return keep
