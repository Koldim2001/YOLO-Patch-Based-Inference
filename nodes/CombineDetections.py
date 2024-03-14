import torch

from nodes.MakeCropsDetectThem import MakeCropsDetectThem


class CombineDetections:
    """
    Class implementing combining masks/boxes from multiple crops + NMS (Non-Maximum Suppression).

    Args:
        element_crops (MakeCropsDetectThem): Object containing crop information.
        nms_threshold (float): IoU/IoS threshold for non-maximum suppression.
        match_metric (str): Matching metric, either 'IOU' or 'IOS'.

    Attributes:
        conf_treshold (float): Confidence threshold of yolov8.
        class_names (dict): Dictionary containing class names pf yolov8 model.
        crops (list): List to store the CropElement objects.
        image (np.ndarray): Source image.
        nms_threshold (float): IoU/IoS threshold for non-maximum suppression.
        match_metric (str): Matching metric (IOU/IOS).
        detected_conf_list_full (list): List of detected confidences.
        detected_xyxy_list_full (list): List of detected bounding boxes.
        detected_masks_list_full (list): List of detected masks.
        detected_cls_id_list_full (list): List of detected class IDs.
        detected_cls_names_list_full (list): List of detected class names.
        filtered_indices (list): List of indices after non-maximum suppression.
        filtered_confidences (list): List of confidences after non-maximum suppression.
        filtered_boxes (list): List of bounding boxes after non-maximum suppression.
        filtered_classes_id (list): List of class IDs after non-maximum suppression.
        filtered_classes_names (list): List of class names after non-maximum suppression.
        filtered_masks (list): List of filtered (after nms) masks if segmentation is enabled.
    """

    def __init__(
        self,
        element_crops: MakeCropsDetectThem,
        nms_threshold=0.3,
        match_metric='IOS'
    ) -> None:
        self.conf_treshold = element_crops.conf
        self.class_names = element_crops.class_names_dict 
        self.crops = element_crops.crops  # List to store the CropElement objects
        if element_crops.resize_initial_size:
            self.image = element_crops.crops[0].source_image
        else:
            self.image = element_crops.crops[0].source_image_resized

        self.nms_threshold = nms_threshold  # IoU treshold for NMS
        self.match_metric = match_metric
        (
            self.detected_conf_list_full,
            self.detected_xyxy_list_full,
            self.detected_masks_list_full,
            self.detected_cls_id_list_full
        ) = self.combinate_detections(crops=self.crops)

        self.detected_cls_names_list_full = [
            self.class_names[value] for value in self.detected_cls_id_list_full
        ]  # make str list

        # Invoke the NMS method for filtering predictions
        self.filtered_indices = self.nms(
            self.detected_conf_list_full,
            self.detected_xyxy_list_full,
            self.match_metric, 
            self.nms_threshold
        )

        # Apply filtering to the prediction lists
        self.filtered_confidences = [self.detected_conf_list_full[i] for i in self.filtered_indices]
        self.filtered_boxes = [self.detected_xyxy_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_id = [self.detected_cls_id_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_names = [self.detected_cls_names_list_full[i] for i in self.filtered_indices]

        if element_crops.segment:
            self.filtered_masks = [self.detected_masks_list_full[i] for i in self.filtered_indices]
        else:
            self.filtered_masks = []

    def combinate_detections(self, crops):
        """
        Combine detections from multiple crop elements.

        Args:
            crops (list): List of CropElement objects.

        Returns:
            tuple: Tuple containing lists of detected confidences, bounding boxes,
                masks, and class IDs.
        """
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

    def nms(self, 
        confidences: list, 
        boxes: list, 
        match_metric,
        nms_threshold,
    ):
        """
        Apply non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.

        Args:
            confidences (list): List of confidence scores.
            boxes (list): List of bounding boxes.
            match_metric (str): Matching metric, either 'IOU' or 'IOS'.
            nms_threshold (float): The threshold for match metric.

        Returns:
            list: List of filtered indexes.
        """

        # Convert lists to tensors
        boxes = torch.tensor(boxes)
        confidences = torch.tensor(confidences)

        # Extract coordinates for every prediction box present
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate area of every box
        areas = (x2 - x1) * (y2 - y1)

        # Sort the prediction boxes according to their confidence scores
        order = confidences.argsort()

        # Initialise an empty list for filtered prediction boxes
        keep = []

        while len(order) > 0:
            # Extract the index of the prediction with highest score
            idx = order[-1]

            # Push the index in filtered predictions list
            keep.append(idx.tolist())

            # Remove the index from the list
            order = order[:-1]

            # If there are no more boxes, break
            if len(order) == 0:
                break

            # Select coordinates of BBoxes according to the indices
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

            # Take max with 0.0 to avoid negative width and height
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)

            # Find the intersection area
            inter = w * h

            # Find the areas of BBoxes
            rem_areas = torch.index_select(areas, dim=0, index=order)

            # Calculate the distance between centers of the boxes
            cx = (x1[idx] + x2[idx]) / 2
            cy = (y1[idx] + y2[idx]) / 2
            rem_cx = (x1[order] + x2[order]) / 2
            rem_cy = (y1[order] + y2[order]) / 2
            dist_centers = ((cx - rem_cx) ** 2 + (cy - rem_cy) ** 2).sqrt()

            if match_metric == "IOU":
                # Find the union of every prediction with the prediction
                union = (rem_areas - inter) + areas[idx]
                # Find the IoU of every prediction
                match_metric_value = inter / union

            elif match_metric == "IOS":
                # Find the smaller area of every prediction with the prediction
                smaller = torch.min(rem_areas, areas[idx])
                # Find the IoU of every prediction
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

            # Keep the boxes with IoU/IoS less than threshold
            mask = match_metric_value < nms_threshold
            order = order[mask]

        return keep
