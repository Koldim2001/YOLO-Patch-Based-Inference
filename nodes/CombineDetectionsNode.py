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
        match_metric = 'IOU', 
        use_greedy_nmm=False
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
        if use_greedy_nmm: 
            self.filtered_indices = self.greedy_nmm(
                self.detected_conf_list_full,
                self.detected_xyxy_list_full,
                self.match_metric,
                self.nms_iou
            )

            for keep_ind, merge_ind_list in self.filtered_indices.items():
                for merge_ind in merge_ind_list:
                    if 
        else:
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
                 # Calculate the distance between centers of the boxes
                # Centers of boxes: (x1 + x2) / 2, (y1 + y2) / 2
                cx = (x1[idx] + x2[idx]) / 2
                cy = (y1[idx] + y2[idx]) / 2
                rem_cx = (x1[order] + x2[order]) / 2
                rem_cy = (y1[order] + y2[order]) / 2
                dist_centers = ((cx - rem_cx) ** 2 + (cy - rem_cy) ** 2).sqrt()

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
    
        
   
    def greedy_nmm(
    self, 
    confidences: list, 
    boxes: list, 
    match_metric,
    iou_threshold,
    ):
        """
        Apply greedy version of non-maximum merging to avoid detecting too many
        overlapping bounding boxes for a given object.
        Args:
            object_predictions_as_tensor: (tensor) The location preds for the image
                along with the class predscores, Shape: [num_boxes,5].
            object_predictions_as_list: ObjectPredictionList Object prediction objects
                to be merged.
            match_metric: (str) IOU or IOS
            match_threshold: (float) The overlap thresh for
                match metric.
        Returns:
            keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
            to keep to a list of prediction indices to be merged.
        """
        keep_to_merge_list = {}

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

        # initialise an empty list for
        # filtered prediction boxes
        keep = []

        while len(order) > 0:
            # extract the index of the
            # prediction with highest score
            # we call this prediction S
            idx = order[-1]

            # push S in filtered predictions list
            keep.append(idx.tolist())

            # remove S from P
            order = order[:-1]

            # sanity check
            if len(order) == 0:
                keep_to_merge_list[idx.tolist()] = []
                break

            # select coordinates of BBoxes according to
            # the indices in order
            xx1 = torch.index_select(x1, dim=0, index=order)
            xx2 = torch.index_select(x2, dim=0, index=order)
            yy1 = torch.index_select(y1, dim=0, index=order)
            yy2 = torch.index_select(y2, dim=0, index=order)

            # find the coordinates of the intersection boxes
            xx1 = torch.max(xx1, x1[idx])
            yy1 = torch.max(yy1, y1[idx])
            xx2 = torch.min(xx2, x2[idx])
            yy2 = torch.min(yy2, y2[idx])

            # find height and width of the intersection boxes
            w = xx2 - xx1
            h = yy2 - yy1

            # take max with 0.0 to avoid negative w and h
            # due to non-overlapping boxes
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            # find the intersection area
            inter = w * h

            # find the areas of BBoxes according the indices in order
            rem_areas = torch.index_select(areas, dim=0, index=order)

            if match_metric == "IOU":
                # find the union of every prediction T in P
                # with the prediction S
                # Note that areas[idx] represents area of S
                union = (rem_areas - inter) + areas[idx]
                # find the IoU of every prediction in P with S
                match_metric_value = inter / union

            elif match_metric == "IOS":
                # find the smaller area of every prediction T in P
                # with the prediction S
                # Note that areas[idx] represents area of S
                smaller = torch.min(rem_areas, areas[idx])
                # find the IoS of every prediction in P with S
                match_metric_value = inter / smaller
            else:
                raise ValueError()

            # keep the boxes with IoU/IoS less than thresh_iou
            mask = match_metric_value < iou_threshold
            matched_box_indices = order[(mask == False).nonzero().flatten()].flip(dims=(0,))
            unmatched_indices = order[(mask == True).nonzero().flatten()]

            # update box pool
            order = unmatched_indices[confidences[unmatched_indices].argsort()]

            # create keep_ind to merge_ind_list mapping
            keep_to_merge_list[idx.tolist()] = []

            for matched_box_ind in matched_box_indices.tolist():
                keep_to_merge_list[idx.tolist()].append(matched_box_ind)

        return keep_to_merge_list

def calculate_bbox_iou_single(box1, box2):
    """
    Calculate Intersection over Union (IoU) for a single pair of bounding boxes.

    Args:
        box1 (list): Coordinates of the first bounding box in the format [x1, y1, x2, y2].
        box2 (list): Coordinates of the second bounding box in the format [x1, y1, x2, y2].

    Returns:
        float: Intersection over Union (IoU) value.
    """
    # Convert bounding box coordinates to [x1, y1, x2, y2] format
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    
    # Calculate intersection and union areas
    intersection_area = x_intersection * y_intersection
    union_area = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def calculate_bbox_ios_single(box1, box2):
    """
    Calculate Intersection over Smaller (IoS) for a single pair of bounding boxes.

    Args:
        box1 (list): Coordinates of the first bounding box in the format [x1, y1, x2, y2].
        box2 (list): Coordinates of the second bounding box in the format [x1, y1, x2, y2].

    Returns:
        float: Intersection over Smaller (IoS) value.
    """
    # Convert bounding box coordinates to [x1, y1, x2, y2] format
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    
    # Calculate intersection and smaller areas
    intersection_area = x_intersection * y_intersection
    smaller_area = min((x2_1 - x1_1) * (y2_1 - y1_1), (x2_2 - x1_2) * (y2_2 - y1_2))
    
    # Calculate IoS
    ios = intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    return ios
