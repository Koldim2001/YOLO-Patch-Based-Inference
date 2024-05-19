import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


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
    return_image_array=False
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
        color_class_background (tuple): The background bgr color for class labels. Default is (0, 0, 255) (red).
        color_class_text (tuple): The text color for class labels. Default is (255, 255, 255) (white).
        thickness (int): The thickness of bounding box and text. Default is 4.
        font: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): The scale factor for font size. Default is 1.5.
        delta_colors (int): The random seed offset for color variation. Default is 0.
        dpi (int): Final visualization size (plot is bigger when dpi is higher).
        random_object_colors (bool): If True, colors for each object are selected randomly.
        show_confidences (bool): If True and show_class=True, confidences near class are visualized.
        axis_off (bool): If True, axis is turned off in the final visualization.
        show_classes_list (list): If empty, visualize all classes. Otherwise, visualize only classes in the list.
        return_image_array (bool): If True, the function returns the image bgr array instead of displaying it. 
                                   Default is False.

    Returns:
        None/np.array
    """

    # Perform inference
    predictions = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    labeled_image = img.copy()

    if random_object_colors:
        random.seed(int(delta_colors))

    # Process each prediction
    for pred in predictions:

        class_names = pred.names

        # Get the bounding boxes and convert them to a list of lists
        boxes = pred.boxes.xyxy.cpu().int().tolist()

        # Get the classes and convert them to a list
        classes = pred.boxes.cls.cpu().int().tolist()

        # Get the mask confidence scores
        confidences = pred.boxes.conf.cpu().numpy()

        num_objects = len(classes)

        if segment:
            # Get the masks
            try:
                masks = pred.masks.data.cpu().numpy()
            except:
                masks = []

        # Visualization
        for i in range(num_objects):
            # Get the class for the current detection
            class_index = int(classes[i])
            class_name = class_names[class_index]

            if show_classes_list and class_index not in show_classes_list:
                continue

            if random_object_colors:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                # Assign color according to class
                random.seed(int(classes[i] + delta_colors))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            box = boxes[i]
            x_min, y_min, x_max, y_max = box

            if segment:
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
                    color_mask = np.zeros_like(img)
                    color_mask[mask_resized > 0] = color
                    labeled_image = cv2.addWeighted(labeled_image, 1, color_mask, alpha, 0)

                cv2.drawContours(labeled_image, mask_contours, -1, color, thickness)

            # Write class label
            if show_boxes:
                cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), color, thickness)

            if show_class:
                if show_confidences:
                    label = f'{str(class_name)} {confidences[i]:.2}'
                else:
                    label = str(class_name)
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(
                    labeled_image,
                    (x_min, y_min),
                    (x_min + text_width + 5, y_min + text_height + 5),
                    color_class_background,
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
        color_class_background (tuple): The background bgr color for class labels. Default is (0, 0, 255) (red).
        color_class_text (tuple): The text color for class labels. Default is (255, 255, 255) (white).
        thickness (int): The thickness of bounding box and text. Default is 4.
        font: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): The scale factor for font size. Default is 1.5.
        delta_colors (int): The random seed offset for color variation. Default is 0.
        dpi (int): Final visualization size (plot is bigger when dpi is higher). Default is 150.
        random_object_colors (bool): If true, colors for each object are selected randomly. Default is False.
        show_confidences (bool): If true and show_class=True, confidences near class are visualized. Default is False.
        axis_off (bool): If true, axis is turned off in the final visualization. Default is True.
        show_classes_list (list): If empty, visualize all classes. Otherwise, visualize only classes in the list.
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
        else:
            # Assign color according to class
            random.seed(int(classes_ids[i] + delta_colors))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

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
                cv2.drawContours(labeled_image, [points], -1, color, thickness)
                if fill_mask:
                    if alpha == 1:
                        cv2.fillPoly(labeled_image, pts=[points], color=color)
                    else:
                        mask_from_poly = np.zeros_like(img)
                        color_mask_from_poly = cv2.fillPoly(mask_from_poly, pts=[points], color=color)
                        labeled_image = cv2.addWeighted(labeled_image, 1, color_mask_from_poly, alpha, 0)

        # Write class label
        if show_boxes:
            cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), color, thickness)

        if show_class:
            if show_confidences:
                label = f'{str(class_name)} {confidences[i]:.2}'
            else:
                label = str(class_name)
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(
                labeled_image,
                (x_min, y_min),
                (x_min + text_width + 5, y_min + text_height + 5),
                color_class_background,
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
