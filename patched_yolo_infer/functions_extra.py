import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


def visualize_results_usual_yolo_inference(
    img,
    model,
    imgsz=640,
    conf=0.5,
    iou=0.7,
    segment=False,
    show_boxes=True,
    show_class=True,
    fill_mask=False,
    alpha=0.3,
    color_class_background=(0, 0, 255),
    color_class_text=(255, 255, 255),
    thickness=4,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.5,
    delta_colors=0,
    dpi=150,
    random_object_colors=False,
    show_confidences=False,
    axis_off=True,
    show_classes_list=[],
    list_of_class_colors=None,
    return_image_array=False,
    inference_extra_args=None,
):
    """
    Visualizes the results of usual YOLOv8 or YOLOv8-seg inference on an image

    Args:
        img (numpy.ndarray): The input image in BGR format.
        model: The object detection or segmentation model (yolov8).
        imgsz (int): The input image size for the model. Default is 640.
        conf (float): The confidence threshold for detection. Default is 0.5.
        iou (float): The intersection over union threshold for detection. Default is 0.7.
        segment (bool): Whether to perform instance segmentation. Default is False.
        show_boxes (bool): Whether to show bounding boxes. Default is True.
        show_class (bool): Whether to show class labels. Default is True.
        fill_mask (bool): Whether to fill the segmented regions with color. Default is False.
        alpha (float): The transparency of filled masks. Default is 0.3.
        color_class_background (tuple / list of tuple): The background BGR color for class labels. Default is (0, 0, 255) (red).
        color_class_text (tuple): The text BGR color for class labels. Default is (255, 255, 255) (white).
        thickness (int): The thickness of bounding box and text. Default is 4.
        font: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): The scale factor for font size. Default is 1.5.
        delta_colors (int): The random seed offset for color variation. Default is 0.
        dpi (int): Final visualization size (plot is bigger when dpi is higher).
        random_object_colors (bool): If True, colors for each object are selected randomly.
        show_confidences (bool): If True and show_class=True, confidences near class are visualized.
        axis_off (bool): If True, axis is turned off in the final visualization.
        show_classes_list (list): If empty, visualize all classes. Otherwise, visualize only classes in the list.
        inference_extra_args (dict/None): Dictionary with extra ultralytics inference parameters.
        list_of_class_colors (list/None): A list of tuples representing the colors for each class in BGR format.  
                    If provided, these colors will be used for displaying the classes instead of random colors. 
                    The number of tuples in the list must match the number of possible classes in the network.
        return_image_array (bool): If True, the function returns the image bgr array instead of displaying it. 
                                   Default is False.

    Returns:
        None/np.array
    """

    # Perform inference
    extra_args = {} if inference_extra_args is None else inference_extra_args
    predictions = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False, **extra_args)

    labeled_image = img.copy()

    if random_object_colors:
        random.seed(int(delta_colors))

    class_names = model.names

    # Process each prediction
    for pred in predictions:

        # Get the bounding boxes and convert them to a list of lists
        boxes = pred.boxes.xyxy.cpu().int().tolist()

        # Get the classes and convert them to a list
        classes = pred.boxes.cls.cpu().int().tolist()

        # Get the mask confidence scores
        confidences = pred.boxes.conf.cpu().numpy()

        num_objects = len(classes)

        if segment:
            # Get the polygons
            try:
                polygons = pred.masks.xy
            except:
                polygons = []

        # Visualization
        for i in range(num_objects):
            # Get the class for the current detection
            class_index = int(classes[i])
            class_name = class_names[class_index]

            if show_classes_list and class_index not in show_classes_list:
                continue

            if random_object_colors:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            elif list_of_class_colors is None:
                # Assign color according to class
                random.seed(int(classes[i] + delta_colors))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = list_of_class_colors[class_index]

            box = boxes[i]
            x_min, y_min, x_max, y_max = box

            if segment and len(polygons) > 0:
                if len(polygons[i]) > 0:
                    points = np.array(polygons[i].reshape((-1, 1, 2)), dtype=np.int32)
                    if fill_mask:
                        if alpha == 1:
                            cv2.fillPoly(labeled_image, pts=[points], color=color)
                        else:
                            mask_from_poly = np.zeros_like(img)
                            color_mask_from_poly = cv2.fillPoly(mask_from_poly, pts=[points], color=color)
                            labeled_image = cv2.addWeighted(labeled_image, 1, color_mask_from_poly, alpha, 0)
                    cv2.drawContours(labeled_image, [points], -1, color, thickness)

            # Write class label
            if show_boxes:
                cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), color, thickness)

            if show_class:
                if show_confidences:
                    label = f'{str(class_name)} {confidences[i]:.2}'
                else:
                    label = str(class_name)
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                background_color = (
                    color_class_background[class_index]
                    if isinstance(color_class_background, list)
                    else color_class_background
                )
                cv2.rectangle(
                    labeled_image,
                    (x_min, y_min),
                    (x_min + text_width + 5, y_min + text_height + 5),
                    background_color,
                    -1,
                )
                cv2.putText(
                    labeled_image,
                    label,
                    (x_min + 5, y_min + text_height),
                    font,
                    font_scale,
                    color_class_text,
                    thickness=thickness,
                )

    if return_image_array:
        return labeled_image
    else:
        # Display the final image with overlaid masks and labels
        plt.figure(figsize=(8, 8), dpi=dpi)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        plt.imshow(labeled_image)
        if axis_off:
            plt.axis('off')
        plt.show()


def get_crops(
    image_full,
    shape_x: int,
    shape_y: int,
    overlap_x=15,
    overlap_y=15,
    show=False,
    save_crops=False,
    save_folder="crops_folder",
    start_name="image",
    resize=False,
):
    """
    Preprocessing of the image. Generating crops with overlapping.

    Args:
        image_full (array): numpy array of a BGR image.
        shape_x (int): size of the crop in the x-coordinate.
        shape_y (int): size of the crop in the y-coordinate.
        overlap_x (float, optional): Percentage of overlap along the x-axis
            (how much subsequent crops borrow information from previous ones). Default is 15.
        overlap_y (float, optional): Percentage of overlap along the y-axis
            (how much subsequent crops borrow information from previous ones). Default is 15.
        show (bool): enables the mode to display images using plt.imshow. Default is False.
        save_crops (bool): enables saving generated images. Default is False.
        save_folder (str): folder path to save the images. Default is "crops_folder".
        start_name (str): starting name for saved images. Default is "image".
        resize (bool): If True, the image is resized to fit the last crop exactly. 
                       If False, the image is not resized. Default is False.

    Returns:
        data_all_crops (list): List containing cropped images.
    """
    
    cross_koef_x = 1 - (overlap_x / 100)
    cross_koef_y = 1 - (overlap_y / 100)

    data_all_crops = []

    y_steps = int((image_full.shape[0] - shape_y) / (shape_y * cross_koef_y)) + 1
    x_steps = int((image_full.shape[1] - shape_x) / (shape_x * cross_koef_x)) + 1

    if resize:
        y_new = round((y_steps-1) * (shape_y * cross_koef_y) + shape_y)
        x_new = round((x_steps-1) * (shape_x * cross_koef_x) + shape_x)
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

            data_all_crops.append(im_temp)

            # Save the image
            if save_crops:
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                filename = f"{save_folder}/{start_name}_crop_{count}.png"
                cv2.imwrite(filename, im_temp)

    if show:
        plt.show()
        print('Number of generated images:', count)

    return data_all_crops


def visualize_results(
    img,
    boxes,
    classes_ids,
    confidences=[],
    classes_names=[], 
    masks=[],
    polygons=[],
    segment=False,
    show_boxes=True,
    show_class=True,
    fill_mask=False,
    alpha=0.3,
    color_class_background=(0, 0, 255),
    color_class_text=(255, 255, 255),
    thickness=4,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.5,
    delta_colors=0,
    dpi=150,
    random_object_colors=False,
    show_confidences=False,
    axis_off=True,
    show_classes_list=[],
    list_of_class_colors=None,
    return_image_array=False
):
    """
    Visualizes custom results of object detection or segmentation on an image.

    Args:
        img (numpy.ndarray): The input image in BGR format.
        boxes (list): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        classes_ids (list): A list of class IDs for each detection.
        confidences (list): A list of confidence scores corresponding to each bounding box. Default is an empty list.
        classes_names (list): A list of class names corresponding to the class IDs. Default is an empty list.
        masks (list): A list of masks. Default is an empty list.
        segment (bool): Whether to perform instance segmentation. Default is False.
        show_boxes (bool): Whether to show bounding boxes. Default is True.
        show_class (bool): Whether to show class labels. Default is True.
        fill_mask (bool): Whether to fill the segmented regions with color. Default is False.
        alpha (float): The transparency of filled masks. Default is 0.3.
        color_class_background (tuple / list of tuple): The background BGR color for class labels. Default is (0, 0, 255) (red).
        color_class_text (tuple): The text BGR color for class labels. Default is (255, 255, 255) (white).
        thickness (int): The thickness of bounding box and text. Default is 4.
        font: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): The scale factor for font size. Default is 1.5.
        delta_colors (int): The random seed offset for color variation. Default is 0.
        dpi (int): Final visualization size (plot is bigger when dpi is higher). Default is 150.
        random_object_colors (bool): If true, colors for each object are selected randomly. Default is False.
        show_confidences (bool): If true and show_class=True, confidences near class are visualized. Default is False.
        axis_off (bool): If true, axis is turned off in the final visualization. Default is True.
        show_classes_list (list): If empty, visualize all classes. Otherwise, visualize only classes in the list.
        list_of_class_colors (list/None): A list of tuples representing the colors for each class in BGR format. 
                    If provided, these colors will be used for displaying the classes instead of random colors. 
                    The number of tuples in the list must match the number of possible classes in the network.
        return_image_array (bool): If True, the function returns the image bgr array instead of displaying it.
                    Default is False.
                                   
    Returns:
        None/np.array
    """

    # Create a copy of the input image
    labeled_image = img.copy()

    if random_object_colors:
        random.seed(int(delta_colors))

    # Process each prediction
    for i in range(len(classes_ids)):
        # Get the class for the current detection
        if len(classes_names)>0:
            class_name = str(classes_names[i])
        else:
            class_name = str(classes_ids[i])

        if show_classes_list and int(classes_ids[i]) not in show_classes_list:
            continue

        if random_object_colors:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif list_of_class_colors is None:
            # Assign color according to class
            random.seed(int(classes_ids[i] + delta_colors))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            color = list_of_class_colors[classes_ids[i]]

        box = boxes[i]
        x_min, y_min, x_max, y_max = box

        if segment and len(masks) > 0:
            mask = masks[i]
            # Resize mask to the size of the original image using nearest neighbor interpolation
            mask_resized = cv2.resize(
                np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            # Add label to the mask
            mask_contours, _ = cv2.findContours(
                mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if fill_mask:
                if alpha == 1:
                    cv2.fillPoly(labeled_image, pts=mask_contours, color=color)
                else:
                    color_mask = np.zeros_like(img)
                    color_mask[mask_resized > 0] = color
                    labeled_image = cv2.addWeighted(labeled_image, 1, color_mask, alpha, 0)
            cv2.drawContours(labeled_image, mask_contours, -1, color, thickness)
        
        elif segment and len(polygons) > 0:
            if len(polygons[i]) > 0:
                points = np.array(polygons[i].reshape((-1, 1, 2)), dtype=np.int32)
                if fill_mask:
                    if alpha == 1:
                        cv2.fillPoly(labeled_image, pts=[points], color=color)
                    else:
                        mask_from_poly = np.zeros_like(img)
                        color_mask_from_poly = cv2.fillPoly(mask_from_poly, pts=[points], color=color)
                        labeled_image = cv2.addWeighted(labeled_image, 1, color_mask_from_poly, alpha, 0)
                cv2.drawContours(labeled_image, [points], -1, color, thickness)

        # Write class label
        if show_boxes:
            cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), color, thickness)

        if show_class:
            if show_confidences:
                label = f'{str(class_name)} {confidences[i]:.2}'
            else:
                label = str(class_name)
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            background_color = (
                color_class_background[classes_ids[i]]
                if isinstance(color_class_background, list)
                else color_class_background
            )
            cv2.rectangle(
                labeled_image,
                (x_min, y_min),
                (x_min + text_width + 5, y_min + text_height + 5),
                background_color,
                -1,
            )
            cv2.putText(
                labeled_image,
                label,
                (x_min + 5, y_min + text_height),
                font,
                font_scale,
                color_class_text,
                thickness=thickness,
            )

    if return_image_array:
        return labeled_image
    else:
        # Display the final image with overlaid masks and labels
        plt.figure(figsize=(8, 8), dpi=dpi)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        plt.imshow(labeled_image)
        if axis_off:
            plt.axis('off')
        plt.show()


def create_masks_from_polygons(polygons, image):
    """
    Create binary masks from a list of polygons.

    This function takes a list of polygons and an image, and generates binary masks
    where each mask corresponds to one polygon. The masks are boolean arrays with
    the same dimensions as the input image, where the regions covered by the polygons
    are marked as True.

    Parameters:
    polygons (list of numpy.ndarray): A list of polygons, where each polygon is
        represented as a numpy array of shape (N, 2) containing N (x, y) coordinates.
    image (numpy.ndarray): The input image, used to determine the dimensions of the masks.

    Returns:
    list of numpy.ndarray: A list of binary masks, where each mask is a boolean
        numpy array of the same dimensions as the input image.
    """
    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Create empty masks
    masks = []
    
    for polygon in polygons:
        if len(polygon) > 0:
            points = np.array(polygon.reshape((-1, 1, 2)), dtype=np.int32)
        
        # Create an empty mask with the same size as the image
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw the polygon on the mask
        cv2.fillPoly(mask, [points], 1)
        
        # Add the mask to the list
        masks.append(mask)
    
    return masks


def basic_crop_size_calculation(width, height):
    """
    Calculate the basic crop size and overlap based on the image dimensions.

    This function determines the optimal crop size and overlap for an image based on its width and height.
    The function uses predefined thresholds to decide the crop size and overlap, ensuring efficient processing
    for different image resolutions.

    Parameters:
    width (int): The width of the image in pixels.
    height (int): The height of the image in pixels.

    Returns:
    tuple: A tuple containing the crop size in the x direction (crop_shape_x), crop size in the y direction
        (crop_shape_y), overlap in the x direction (crop_overlap_x), and overlap in the y direction (crop_overlap_y).
    """
    total_pixels = width * height

    if total_pixels <= 640*480:
        crop_shape_x, crop_shape_y = width, height
        crop_overlap_x, crop_overlap_y = 0, 0
    elif total_pixels < 720*576:
        crop_shape_x, crop_shape_y = int(width / 1.25), int(height / 1.25)
        crop_overlap_x, crop_overlap_y = 50, 50
    elif total_pixels < 1920*1080:
        crop_shape_x, crop_shape_y = width // 2, height // 2
        crop_overlap_x, crop_overlap_y = 40, 40
    elif total_pixels < 3840*2160:
        crop_shape_x, crop_shape_y = width // 3, height // 3
        crop_overlap_x, crop_overlap_y = 30, 30
    elif total_pixels < 7680*4320:
        crop_shape_x, crop_shape_y = width // 4, height // 4
        crop_overlap_x, crop_overlap_y = 25, 25
    else:
        crop_shape_x, crop_shape_y = width // 5, height // 5
        crop_overlap_x, crop_overlap_y = 25, 25

    return crop_shape_x, crop_shape_y, crop_overlap_x, crop_overlap_y


def auto_calculate_crop_values(image, mode="network_based", model=None, classes_list=None, conf=0.25):
    """
    Automatically calculate the optimal crop size and overlap for an image.

    This function determines the optimal crop size and overlap for an image based on either the image size
    or the detected objects within the image. The function can use a YOLO model to detect objects and adjust
    the crop size and overlap accordingly.

    Parameters:
    image (numpy.ndarray): The input BGR image.
    mode (str): The type of analysis to perform. Can be "resolution_based" or "network_based". 
        Default is "network_analysis".
    model (YOLO): The YOLO model to use for object detection. If None, a default model yolov8m
        will be loaded. Default is None.
    classes_list (list): A list of class indices to consider for object detection. If None, all classes 
        will be considered. Default is None.
    conf (float): The confidence threshold for detection in "network_based" mode. Default is 0.25.

    Returns:
    tuple: A tuple containing the crop size in the x direction (crop_shape_x), crop size in the y direction 
        (crop_shape_y), overlap in the x direction (crop_overlap_x), and overlap in the y direction (crop_overlap_y).
    """
    height, width = image.shape[:2]

    # If the mode is 'image_size_analysis', calculate crop size based on image dimensions
    if mode == 'resolution_based':
        crop_shape_x, crop_shape_y, crop_overlap_x, crop_overlap_y = basic_crop_size_calculation(
            width, height
        )
    else:
        # If no model is provided, load a default YOLO model
        if model is None:
            model = YOLO("yolov8m.pt")

        # Perform object detection on the image
        result = model.predict(image, conf=conf, iou=0.75, classes=classes_list, verbose=False)

        # If no objects are detected, calculate crop size based on image dimensions
        if len(result[0].boxes) == 0:
            crop_shape_x, crop_shape_y, crop_overlap_x, crop_overlap_y = (
                basic_crop_size_calculation(width, height)
            )
            return crop_shape_x, crop_shape_y, crop_overlap_x, crop_overlap_y

        max_width = 0
        max_height = 0

        # Iterate through detected boxes to find the maximum width and height
        for box in result[0].boxes:
            _, _, box_width, box_height = box.xywh[0].tolist()  
            if box_width > max_width:
                max_width = box_width
            if box_height > max_height:
                max_height = box_height

        # Determine the maximum dimension (width or height) of the detected objects
        max_value = max(max_width, max_height)

        # Adjust crop size and overlap based on the maximum detected object dimension
        if width > height:
            crop_shape_x = int(max_value * 3)  
            crop_shape_y = int(max_value * 2)
        elif width < height:
            crop_shape_x = int(max_value * 2)  
            crop_shape_y = int(max_value * 3)
        else:
            crop_shape_x = int(max_value * 2.5)  
            crop_shape_y = int(max_value * 2.5)

        crop_overlap_x = int(max_width/crop_shape_x * 1.2 * 100)
        crop_overlap_y = int(max_height/crop_shape_y * 1.2 * 100)

        # Ensure the overlap does not exceed 70%
        if crop_overlap_x > 70:
            crop_overlap_x = 70
        if crop_overlap_y > 70:
            crop_overlap_y = 70    

        # Ensure the number of crops does not exceed 7 in each direction
        if height // crop_shape_y > 7:
            crop_shape_y = height // 7
        if width // crop_shape_x > 7:
            crop_shape_x = width // 7

        # Handling cases where patches are not needed along the axes
        if height / crop_shape_y < 1.25:
            crop_shape_y = height
            crop_overlap_y = 0
        if width / crop_shape_x < 1.25:
            crop_shape_x = width
            crop_overlap_x = 0

    return crop_shape_x, crop_shape_y, crop_overlap_x, crop_overlap_y
