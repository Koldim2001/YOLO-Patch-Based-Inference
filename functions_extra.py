import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def visualize_results_usual_yolo_inference(img, model, imgsz=640, conf=0.5, iou=0.7, segment=False, show_boxes=True,
     show_class=True, fill_mask=False, alpha=0.3, color_class_background=(0, 0, 255), color_class_text=(255, 255, 255),
     thickness=4, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.5, delta_colors=0, dpi=150):
    """
    Visualizes the results of object detection or segmentation on an image.

    Args:
        img (numpy.ndarray): The input image.
        model: The object detection or segmentation model.
        imgsz (int): The input image size for the model. Default is 640.
        conf (float): The confidence threshold for detection. Default is 0.5.
        iou (float): The intersection over union threshold for detection. Default is 0.7.
        segment (bool): Whether to perform instance segmentation. Default is False.
        show_boxes (bool): Whether to show bounding boxes. Default is True.
        show_class (bool): Whether to show class labels. Default is True.
        fill_mask (bool): Whether to fill the segmented regions with color. Default is False.
        alpha (float): The transparency of filled masks. Default is 0.3.
        color_class_background (tuple): The background color for class labels. Default is (0, 0, 255) (blue).
        color_class_text (tuple): The text color for class labels. Default is (255, 255, 255) (white).
        thickness (int): The thickness of bounding box and text. Default is 4.
        font: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): The scale factor for font size. Default is 1.5.
        delta_colors (int): The random seed offset for color variation. Default is 0.
        dpi (int): Final visualisation size (plot is bigger when dpi is higher)

    Returns:
        None

    """

    # Perform inference
    predictions = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    labeled_image = img.copy()

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
            masks = pred.masks.data.cpu().numpy()

        # Visualization
        for i in range(num_objects):
            # Get the class for the current detection
            class_index = int(classes[i])
            class_name = class_names[class_index]

            # Assign color according to class
            random.seed(int(classes[i] + delta_colors))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            box = boxes[i]
            x_min, y_min, x_max, y_max = box

            if segment:
                mask = masks[i]
                # Resize mask to the size of the original image using nearest neighbor interpolation
                mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Add label to the mask
                mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(labeled_image, mask_contours, -1, color, thickness)

                if fill_mask:
                    color_mask = np.zeros_like(img)
                    color_mask[mask_resized > 0] = color
                    labeled_image = cv2.addWeighted(labeled_image, 1, color_mask, alpha, 0)

            # Write class label
            if show_boxes:
                cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), color, thickness)

            if show_class:
                label = str(class_name)
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(labeled_image, (x_min, y_min), (x_min + text_width + 5, y_min + text_height + 5),
                              color_class_background, -1)
                cv2.putText(labeled_image, label, (x_min + 5, y_min + text_height), font, font_scale, color_class_text,
                            thickness=thickness)

        # Display the final image with overlaid masks and labels
        plt.figure(figsize=(8, 8), dpi=dpi)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        plt.imshow(labeled_image)
        plt.axis('off')
        plt.show()




def get_crops(image_full, shape_x: int, shape_y: int,
                 overlap_x=15, overlap_y=15, show=False):
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
                plt.imshow(im_temp)
                plt.axis('off')
            count += 1

            data_all_crops.append(im_temp)
    if show:
        plt.show()
        print('Number of generated images:', count)

    return data_all_crops




def visualize_results(
    img,
    confidences,
    boxes,
    classes_ids,
    classes_names=[], 
    masks=[],
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
    dpi=150
):
    """
    Visualizes the results of object detection or segmentation on an image.

    Args:
        img (numpy.ndarray): The input image.
        confidences (list): A list of confidence scores corresponding to each bounding box.
        boxes (list): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        masks (list): A list of masks.
        classes_ids (list): A list of class IDs for each detection.
        classes_names (list): A list of class names corresponding to the class IDs.
        segment (bool): Whether to perform instance segmentation. Default is False.
        show_boxes (bool): Whether to show bounding boxes. Default is True.
        show_class (bool): Whether to show class labels. Default is True.
        fill_mask (bool): Whether to fill the segmented regions with color. Default is False.
        alpha (float): The transparency of filled masks. Default is 0.3.
        color_class_background (tuple): The background color for class labels. Default is (0, 0, 255) (blue).
        color_class_text (tuple): The text color for class labels. Default is (255, 255, 255) (white).
        thickness (int): The thickness of bounding box and text. Default is 4.
        font: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float): The scale factor for font size. Default is 1.5.
        delta_colors (int): The random seed offset for color variation. Default is 0.
        dpi (int): Final visualization size (plot is bigger when dpi is higher). Default is 150.

    Returns:
        None
    """
    
    # Create a copy of the input image
    labeled_image = img.copy()
    
    # Process each prediction
    for i in range(len(confidences)):
        # Get the class for the current detection
        if len(classes_names)>0:
            class_name = str(classes_names[i])
        else:
            class_name = str(classes_ids[i])

        # Assign color according to class
        random.seed(int(classes_ids[i] + delta_colors))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        box = boxes[i]
        x_min, y_min, x_max, y_max = box

        if segment:
            mask = masks[i]
            # Resize mask to the size of the original image using nearest neighbor interpolation
            mask_resized = cv2.resize(np.array(mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Add label to the mask
            mask_contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(labeled_image, mask_contours, -1, color, thickness)

            if fill_mask:
                color_mask = np.zeros_like(img)
                color_mask[mask_resized > 0] = color
                labeled_image = cv2.addWeighted(labeled_image, 1, color_mask, alpha, 0)

        # Write class label
        if show_boxes:
            cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), color, thickness)

        if show_class:
            label = str(class_name)
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(labeled_image, (x_min, y_min), (x_min + text_width + 5, y_min + text_height + 5),
                          color_class_background, -1)
            cv2.putText(labeled_image, label, (x_min + 5, y_min + text_height), font, font_scale, color_class_text,
                        thickness=thickness)

    # Display the final image with overlaid masks and labels
    plt.figure(figsize=(8, 8), dpi=dpi)
    labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
    plt.imshow(labeled_image)
    plt.axis('off')
    plt.show()
