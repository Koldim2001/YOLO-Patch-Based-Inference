import torch
import numpy as np
from .MakeCropsDetectThem import MakeCropsDetectThem


class CombineDetections:
    """
    Class implementing combining masks/boxes from multiple crops + NMS (Non-Maximum Suppression).

    Args:
        element_crops (MakeCropsDetectThem): Object containing crop information.
        nms_threshold (float): IoU/IoS threshold for non-maximum suppression.  Dafault is 0.3.
        match_metric (str): Matching metric, either 'IOU' or 'IOS'. Dafault is IoS.
        class_agnostic_nms (bool) Determines the NMS mode in object detection. When set to True, NMS 
            operates across all classes, ignoring class distinctions and suppressing less confident 
            bounding boxes globally. Otherwise, NMS is applied separately for each class. Default is True.
        intelligent_sorter (bool): Enable sorting by area and rounded confidence parameter. 
            If False, sorting will be done only by confidence (usual nms). Dafault is True.
        sorter_bins (int): Number of bins to use for intelligent_sorter. A smaller number of bins makes
            the NMS more reliant on object sizes rather than confidence scores. Defaults to 5.

    Attributes:
        class_names (dict): Dictionary containing class names of yolo model.
        crops (list): List to store the CropElement objects.
        image (np.ndarray): Source image in BGR.
        nms_threshold (float): IOU/IOS threshold for non-maximum suppression.
        match_metric (str): Matching metric (IOU/IOS).
        class_agnostic_nms (bool) Determines the NMS mode in object detection.
        intelligent_sorter (bool): Flag indicating whether sorting by area and confidence parameter is enabled.
        sorter_bins (int): Number of bins to use for intelligent_sorter. 
        detected_conf_list_full (list): List of detected confidences.
        detected_xyxy_list_full (list): List of detected bounding boxes.
        detected_masks_list_full (list): List of detected masks.
        detected_polygons_list_full (list): List of detected polygons when memory optimization is enabled.
        detected_cls_id_list_full (list): List of detected class IDs.
        detected_cls_names_list_full (list): List of detected class names.
        filtered_indices (list): List of indices after non-maximum suppression.
        filtered_confidences (list): List of confidences after non-maximum suppression.
        filtered_boxes (list): List of bounding boxes after non-maximum suppression.
        filtered_classes_id (list): List of class IDs after non-maximum suppression.
        filtered_classes_names (list): List of class names after non-maximum suppression.
        filtered_masks (list): List of filtered (after nms) masks if segmentation is enabled.
        filtered_polygons (list): List of filtered (after nms) polygons if segmentation and
            memory optimization are enabled.
    """

    def __init__(
        self,
        element_crops: MakeCropsDetectThem,
        nms_threshold=0.3,
        match_metric='IOS',
        intelligent_sorter=True,
        sorter_bins=5,
        class_agnostic_nms=True
    ) -> None:
        self.class_names = element_crops.class_names_dict 
        self.crops = element_crops.crops  # List to store the CropElement objects
        if element_crops.resize_initial_size:
            self.image = element_crops.crops[0].source_image
        else:
            self.image = element_crops.crops[0].source_image_resized

        self.nms_threshold = nms_threshold  # IOU or IOS treshold for NMS
        self.match_metric = match_metric 
        self.intelligent_sorter = intelligent_sorter # enable sorting by area and confidence parameter
        self.sorter_bins = sorter_bins
        self.class_agnostic_nms = class_agnostic_nms

        # Combinate detections of all patches
        (
            self.detected_conf_list_full,
            self.detected_xyxy_list_full,
            self.detected_masks_list_full,
            self.detected_cls_id_list_full,
            self.detected_polygons_list_full
        ) = self.combinate_detections(crops=self.crops)

        self.detected_cls_names_list_full = [
            self.class_names[value] for value in self.detected_cls_id_list_full
        ]  # make str list

        # Invoke the NMS:
        if self.class_agnostic_nms:
            self.filtered_indices = self.nms(
                torch.tensor(self.detected_conf_list_full),
                torch.tensor(self.detected_xyxy_list_full),
                self.match_metric,
                self.nms_threshold,
                self.detected_masks_list_full,
                intelligent_sorter=self.intelligent_sorter
            ) 

        else:
            self.filtered_indices = self.not_agnostic_nms(
                torch.tensor(self.detected_cls_id_list_full),
                torch.tensor(self.detected_conf_list_full),
                torch.tensor(self.detected_xyxy_list_full),
                self.match_metric,
                self.nms_threshold,
                self.detected_masks_list_full,
                intelligent_sorter=self.intelligent_sorter
            )  

        # Apply filtering (nms output indeces) to the prediction lists
        self.filtered_confidences = [self.detected_conf_list_full[i] for i in self.filtered_indices]
        self.filtered_boxes = [self.detected_xyxy_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_id = [self.detected_cls_id_list_full[i] for i in self.filtered_indices]
        self.filtered_classes_names = [self.detected_cls_names_list_full[i] for i in self.filtered_indices]

        # Masks filtering:
        if element_crops.segment and not element_crops.memory_optimize:
            self.filtered_masks = [self.detected_masks_list_full[i] for i in self.filtered_indices]
        else:
            self.filtered_masks = []

        # Polygons filtering:
        if element_crops.segment and element_crops.memory_optimize:
            self.filtered_polygons = [self.detected_polygons_list_full[i] for i in self.filtered_indices]
        else:
            self.filtered_polygons = []

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
        detected_polygons = []

        for crop in crops:
            detected_conf.extend(crop.detected_conf)
            detected_xyxy.extend(crop.detected_xyxy_real)
            detected_masks.extend(crop.detected_masks_real)
            detected_cls.extend(crop.detected_cls)
            detected_polygons.extend(crop.detected_polygons_real)

        return detected_conf, detected_xyxy, detected_masks, detected_cls, detected_polygons

    @staticmethod
    def average_to_bound(confidences, N=10):
        """
        Bins the given confidences into N equal intervals between 0 and 1, 
        and rounds each confidence to the left boundary of the corresponding bin.

        Parameters:
        confidences (list or np.array): List of confidence values to be binned.
        N (int, optional): Number of bins to use. Defaults to 10.

        Returns:
        list: List of rounded confidence values, each bound to the left boundary of its bin.
        """
        # Create the bounds
        step = 1 / N
        bounds = np.arange(0, 1 + step, step)

        # Use np.digitize to determine the corresponding bin for each value
        indices = np.digitize(confidences, bounds, right=True) - 1

        # Bind values to the left boundary of the corresponding bin
        averaged_confidences = np.round(bounds[indices], 2) 

        return averaged_confidences.tolist()

    @staticmethod
    def intersect_over_union(mask, masks_list):
        """
        Compute Intersection over Union (IoU) scores for a given mask against a list of masks.

        Args:
            mask (np.ndarray): Binary mask to compare.
            masks_list (list of np.ndarray): List of binary masks for comparison.

        Returns:
            torch.Tensor: IoU scores for each mask in masks_list compared to the input mask.
        """
        iou_scores = []
        for other_mask in masks_list:
            # Compute intersection and union
            intersection = np.logical_and(mask, other_mask).sum()
            union = np.logical_or(mask, other_mask).sum()
            # Compute IoU score, avoiding division by zero
            iou = intersection / union if union != 0 else 0
            iou_scores.append(iou)
        return torch.tensor(iou_scores)

    @staticmethod
    def intersect_over_smaller(mask, masks_list):
        """
        Compute Intersection over Smaller area scores for a given mask against a list of masks.

        Args:
            mask (np.ndarray): Binary mask to compare.
            masks_list (list of np.ndarray): List of binary masks for comparison.

        Returns:
            torch.Tensor: IoU scores for each mask in masks_list compared to the input mask,
                calculated over the smaller area.
        """
        ios_scores = []
        for other_mask in masks_list:
            # Compute intersection and area of smaller mask
            intersection = np.logical_and(mask, other_mask).sum()
            smaller_area = min(mask.sum(), other_mask.sum())
            # Compute IoU score over smaller area, avoiding division by zero
            ios = intersection / smaller_area if smaller_area != 0 else 0
            ios_scores.append(ios)
        return torch.tensor(ios_scores)

    def nms(
        self,
        confidences: torch.tensor,
        boxes: torch.tensor,
        match_metric,
        nms_threshold,
        masks=[],
        intelligent_sorter=False, 
        cls_indexes=None 
    ):
        """
        Apply class-agnostic non-maximum suppression to avoid detecting too many
        overlapping bounding boxes for a given object.

        Args:
            confidences (torch.Tensor): List of confidence scores.
            boxes (torch.Tensor): List of bounding boxes.
            match_metric (str): Matching metric, either 'IOU' or 'IOS'.
            nms_threshold (float): The threshold for match metric.
            masks (list): List of masks. 
            intelligent_sorter (bool, optional): intelligent sorter 
            cls_indexes (torch.Tensor):  indexes from network detections corresponding
                to the defined class,  uses in case of not agnostic nms

        Returns:
            list: List of filtered indexes.
        """
        if len(boxes) == 0:
            return []

        # Extract coordinates for every prediction box present
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # Calculate area of every box
        areas = (x2 - x1) * (y2 - y1)

        # Sort the prediction boxes according to their confidence scores or intelligent_sorter mode
        if intelligent_sorter:
            # Sort the prediction boxes according to their round confidence scores and area sizes
            order = torch.tensor(
                sorted(
                    range(len(confidences)),
                    key=lambda k: (
                        self.average_to_bound(confidences[k].item(), self.sorter_bins),
                        areas[k],
                    ),
                    reverse=False,
                )
            )
        else:
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

            else:
                raise ValueError("Unknown matching metric")

            # If masks are provided and IoU based on bounding boxes is greater than 0,
            # calculate IoU for masks and keep the ones with IoU < nms_threshold
            if len(masks) > 0 and torch.any(match_metric_value > 0):

                mask_mask = match_metric_value > 0 

                order_2 = order[mask_mask]
                filtered_masks = [masks[i] for i in order_2]

                if match_metric == "IOU":
                    mask_iou = self.intersect_over_union(masks[idx], filtered_masks)
                    mask_mask = mask_iou > nms_threshold

                elif match_metric == "IOS":
                    mask_ios = self.intersect_over_smaller(masks[idx], filtered_masks)
                    mask_mask = mask_ios > nms_threshold
                # create a tensor of indences to delete in tensor order
                order_2 = order_2[mask_mask]
                inverse_mask = ~torch.isin(order, order_2)

                # Keep only those order values that are not contained in order_2
                order = order[inverse_mask]

            else:
                # Keep the boxes with IoU/IoS less than threshold
                mask = match_metric_value < nms_threshold

                order = order[mask]
        if cls_indexes is not None:
            keep = [cls_indexes[i] for i in keep]
        return keep

    def not_agnostic_nms(
            self,
            detected_cls_id_list_full,
            detected_conf_list_full, 
            detected_xyxy_list_full, 
            match_metric, 
            nms_threshold, 
            detected_masks_list_full, 
            intelligent_sorter
                     ):
        '''
            Performs Non-Maximum Suppression (NMS) in a non-agnostic manner, where NMS 
            is applied separately to each class.

            Args:
                detected_cls_id_list_full (torch.Tensor): tensor containing the class IDs for each detected object.
                detected_conf_list_full (torch.Tensor):  tensor of confidence scores.
                detected_xyxy_list_full (torch.Tensor): tensor of bounding boxes.
                match_metric (str): Matching metric, either 'IOU' or 'IOS'.
                nms_threshold (float): the threshold for match metric.
                detected_masks_list_full (torch.Tensor):  List of masks. 
                intelligent_sorter (bool, optional): intelligent sorter 

            Returns:
                List[int]: A list of indices representing the detections that are kept after applying
                    NMS for each class separately.

            Notes:
                - This method performs NMS separately for each class, which helps in
                    reducing false positives within each class.
                - If in your scenario, an object of one class can physically be inside
                    an object of another class, you should definitely use this non-agnostic nms
            '''
        all_keeps = []
        for cls in torch.unique(detected_cls_id_list_full):
            cls_indexes = torch.where(detected_cls_id_list_full==cls)[0]
            if len(detected_masks_list_full) > 0:
                masks_of_class = [detected_masks_list_full[i] for i in cls_indexes]
            else:
                masks_of_class = []
            keep_indexes = self.nms(
                    detected_conf_list_full[cls_indexes],
                    detected_xyxy_list_full[cls_indexes],
                    match_metric,
                    nms_threshold,
                    masks_of_class,
                    intelligent_sorter,
                    cls_indexes
                )
            all_keeps.extend(keep_indexes)
        return all_keeps
