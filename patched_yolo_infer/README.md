# YOLO-Patch-Based-Inference

This Python library simplifies SAHI-like inference for instance segmentation tasks, enabling the detection of small objects in images. It caters to both object detection and instance segmentation tasks, supporting a wide range of Ultralytics models. 

The library also provides a sleek customization of the visualization of the inference results for all models, both in the standard approach (direct network run) and the unique patch-based variant.

**Model Support**: The library offers support for multiple ultralytics deep learning [models](https://docs.ultralytics.com/models/), such as YOLOv8, YOLOv8-seg, YOLOv9, YOLOv9-seg, YOLOv10, FastSAM, and RTDETR. Users can select from pre-trained options or utilize custom-trained models to best meet their task requirements.


## Installation
You can install the library via pip:

```bash
pip install patched-yolo-infer
```

Note: If CUDA support is available, it's recommended to pre-install PyTorch with CUDA support before installing the library. Otherwise, the CPU version will be installed by default.


</details>

## Notebooks

Interactive notebooks are provided to showcase the functionality of the library. These notebooks cover batch inference procedures for detection, instance segmentation, custom visualization of inference, and more. Each notebook is paired with a tutorial on YouTube, making it easy to learn and implement features. Check the GitHub page for the current links to the videos: https://github.com/Koldim2001/YOLO-Patch-Based-Inference

__Check this Colab examples:__
                         
Patch-Based-Inference Example - [**Open in Colab**](https://colab.research.google.com/drive/1XCpIYLMFEmGSO0XCOkSD7CcD9SFHSJPA?usp=sharing)

Example of using various functions for visualizing basic YOLOv8/v9 inference results - [**Open in Colab**](https://colab.research.google.com/drive/1eM4o1e0AUQrS1mLDpcgK9HKInWEvnaMn?usp=sharing)


## Usage

### 1. Patch-Based-Inference
To carry out patch-based inference of YOLO models using our library, you need to follow a sequential procedure. First, you create an instance of the `MakeCropsDetectThem` class, providing all desired parameters related to YOLO inference and the patch segmentation principle.<br/> Subsequently, you pass the obtained object of this class to `CombineDetections`, which facilitates the consolidation of all predictions from each overlapping crop, followed by intelligent suppression of duplicates. <br/>Upon completion, you receive the result, from which you can extract the desired outcome of frame processing.

The output obtained from the process includes several attributes that can be leveraged for further analysis or visualization:

1. img: This attribute contains the original image on which the inference was performed. It provides context for the detected objects.

2. confidences: This attribute holds the confidence scores associated with each detected object. These scores indicate the model's confidence level in the accuracy of its predictions.

3. boxes: These bounding boxes are represented as a list of lists, where each list contains four values: [x_min, y_min, x_max, y_max]. These values correspond to the coordinates of the top-left and bottom-right corners of each bounding box.

4. polygons: If available, this attribute provides a list containing NumPy arrays of polygon coordinates that represent segmentation masks corresponding to the detected objects. These polygons can be utilized to accurately outline the boundaries of each object.

5. classes_ids: This attribute contains the class IDs assigned to each detected object. These IDs correspond to specific object classes defined during the model training phase.

6. classes_names: These are the human-readable names corresponding to the class IDs. They provide semantic labels for the detected objects, making the results easier to interpret.

```python
import cv2
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections

# Load the image 
img_path = "test_image.jpg"
img = cv2.imread(img_path)

element_crops = MakeCropsDetectThem(
    image=img,
    model_path="yolov8m.pt",
    segment=False,
    shape_x=640,
    shape_y=640,
    overlap_x=50,
    overlap_y=50,
    conf=0.5,
    iou=0.7,
)
result = CombineDetections(element_crops, nms_threshold=0.25)  

# Final Results:
img=result.image
confidences=result.filtered_confidences
boxes=result.filtered_boxes
polygons=result.filtered_polygons
classes_ids=result.filtered_classes_id
classes_names=result.filtered_classes_names
```

#### Explanation of possible input arguments:

**MakeCropsDetectThem**
Class implementing cropping and passing crops through a neural network for detection/segmentation.\
**Args:**
- **image** (*np.ndarray*): Input image BGR.
- **model_path** (*str*): Path to the YOLO model.
- **model** (*ultralytics model*) Pre-initialized model object. If provided, the model will be used directly instead of loading from model_path.
- **imgsz** (*int*): Size of the input image for inference YOLO.
- **conf** (*float*): Confidence threshold for detections YOLO.
- **iou** (*float*): IoU threshold for non-maximum suppression YOLOv8 of single crop.
- **classes_list** (*List[int] or None*): List of classes to filter detections. If None, all classes are considered. Defaults to None.
- **segment** (*bool*): Whether to perform segmentation (YOLOv8-seg).
- **shape_x** (*int*): Size of the crop in the x-coordinate.
- **shape_y** (*int*): Size of the crop in the y-coordinate.
- **overlap_x** (*float*): Percentage of overlap along the x-axis.
- **overlap_y** (*float*): Percentage of overlap along the y-axis.
- **show_crops** (*bool*): Whether to visualize the cropping.
- **resize_initial_size** (*bool*): Whether to resize the results to the original image size (ps: slow operation).
- **memory_optimize** (*bool*): Memory optimization option for segmentation (less accurate results when enabled).
- **inference_extra_args** (*dict*): Dictionary with extra ultralytics [inference parameters](https://docs.ultralytics.com/modes/predict/#inference-arguments) (possible keys: half, device, max_det, augment, agnostic_nms and retina_masks)
- **batch_inference** (*bool*): Batch inference of image crops through a neural network instead of sequential passes of crops (ps: faster inference, higher gpu memory use)

**CombineDetections**
Class implementing combining masks/boxes from multiple crops + NMS (Non-Maximum Suppression).\
**Args:**
- **element_crops** (*MakeCropsDetectThem*): Object containing crop information. This can be either a single MakeCropsDetectThem object or a list of objects. 
- **nms_threshold** (*float*): IoU/IoS threshold for non-maximum suppression.
- **match_metric** (*str*): Matching metric, either 'IOU' or 'IOS'.
- **class_agnostic_nms** (*bool*) Determines the NMS mode in object detection. When set to True, NMS operates across all classes, ignoring class distinctions and suppressing less confident bounding boxes globally. Otherwise, NMS is applied separately for each class. (Default is True)
- **intelligent_sorter** (*bool*): Enable sorting by area and rounded confidence parameter. If False, sorting will be done only by confidence (usual nms). (Dafault is True)
- **sorter_bins** (*int*): Number of bins to use for intelligent_sorter. A smaller number of bins makes the NMS more reliant on object sizes rather than confidence scores. (Defaults to 5)


---
### 2. Custom inference visualization:
Visualizes custom results of object detection or segmentation on an image.

**Args:**
- **img** (*numpy.ndarray*): The input image in BGR format.
- **boxes** (*list*): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
- **classes_ids** (*list*): A list of class IDs for each detection.
- **confidences** (*list*): A list of confidence scores corresponding to each bounding box. Default is an empty list.
- **classes_names** (*list*): A list of class names corresponding to the class IDs. Default is an empty list.
- **polygons** (*list*): A list containing NumPy arrays of polygon coordinates that represent segmentation masks.
- **masks** (*list*): A list of masks. Default is an empty list.
- **segment** (*bool*): Whether to perform instance segmentation. Default is False.
- **show_boxes** (*bool*): Whether to show bounding boxes. Default is True.
- **show_class** (*bool*): Whether to show class labels. Default is True.
- **fill_mask** (*bool*): Whether to fill the segmented regions with color. Default is False.
- **alpha** (*float*): The transparency of filled masks. Default is 0.3.
- **color_class_background** (*tuple/list*): The background BGR color for class labels. If you pass a list of tuples, then each class will have its own color. Default is (0, 0, 255) (red).
- **color_class_text** (*tuple*): The text color for class labels. Default is (255, 255, 255) (white).
- **thickness** (*int*): The thickness of bounding box and text. Default is 4.
- **font**: The font type for class labels. Default is cv2.FONT_HERSHEY_SIMPLEX.
- **font_scale** (*float*): The scale factor for font size. Default is 1.5.
- **delta_colors** (*int*): The random seed offset for color variation. Default is seed=0.
- **dpi** (*int*): Final visualization size (plot is bigger when dpi is higher). Default is 150.
- **random_object_colors** (*bool*): If true, colors for each object are selected randomly. Default is False.
- **show_confidences** (*bool*): If true and show_class=True, confidences near class are visualized. Default is False.
- **axis_off** (*bool*): If true, axis is turned off in the final visualization. Default is True.
- **show_classes_list** (*list*): If empty, visualize all classes. Otherwise, visualize only classes in the list.
- **list_of_class_colors** (*list*) A list of tuples representing the colors for each class in BGR format. If provided, these colors will be used for displaying the classes instead of random colors.
- **return_image_array** (*bool*): If True, the function returns the image (BGR np.array) instead of displaying it. Default is False.


Example of using:
```python
from patched_yolo_infer import visualize_results

# Assuming result is an instance of the CombineDetections class
result = CombineDetections(...) 

# Visualizing the results using the visualize_results function
visualize_results(
    img=result.image,
    confidences=result.filtered_confidences,
    boxes=result.filtered_boxes,
    polygons=result.filtered_polygons,
    classes_ids=result.filtered_classes_id,
    classes_names=result.filtered_classes_names,
    segment=False,
)
```

---

## __How to improve the quality of the algorithm for the task of instance segmentation:__

In this approach, all operations under the hood are performed on binary masks of recognized objects. Storing these masks consumes a lot of memory, so this method requires more RAM and slightly more processing time. However, the accuracy of recognition significantly improves, which is especially noticeable in cases where there are many objects of different sizes and they are densely packed. Therefore, we recommend using this approach in production if accuracy is important and not speed, and if your computational resources allow storing hundreds of binary masks in RAM.

The difference in the approach to using the function lies in specifying the parameter ```memory_optimize=False``` in the ```MakeCropsDetectThem``` class.
In such a case, the informative values after processing will be the following:

1. img: This attribute contains the original image on which the inference was performed. It provides context for the detected objects.

2. confidences: This attribute holds the confidence scores associated with each detected object. These scores indicate the model's confidence level in the accuracy of its predictions.

3. boxes: These bounding boxes are represented as a list of lists, where each list contains four values: [x_min, y_min, x_max, y_max]. These values correspond to the coordinates of the top-left and bottom-right corners of each bounding box.

4. masks: This attribute provides segmentation binary masks corresponding to the detected objects. These masks can be used to precisely delineate object boundaries.

5. classes_ids: This attribute contains the class IDs assigned to each detected object. These IDs correspond to specific object classes defined during the model training phase.

6. classes_names: These are the human-readable names corresponding to the class IDs. They provide semantic labels for the detected objects, making the results easier to interpret.


Here's how you can obtain them:
```python
img=result.image
confidences=result.filtered_confidences
boxes=result.filtered_boxes
masks=result.filtered_masks
classes_ids=result.filtered_classes_id
classes_names=result.filtered_classes_names
```

---

## __How to automatically determine optimal parameters for patches (crops):__

To efficiently process a large number of images of varying sizes and contents, manually selecting the optimal patch sizes and overlaps can be difficult. To address this, an algorithm has been developed to automatically calculate the best parameters for patches (crops).

The  `auto_calculate_crop_values` function operates in two modes:

1. **Resolution-Based Analysis**: This mode evaluates the resolution of the source images to determine the optimal patch sizes and overlaps. It is faster but may not yield the highest quality results because it does not take into account the actual objects present in the images.

2. **Neural Network-Based Analysis**: This advanced mode employs a neural network to analyze the images. The algorithm performs a standard inference of the network on the entire image and identifies the largest detected objects. Based on the sizes of these objects, the algorithm selects patch parameters to ensure that the largest objects are fully contained within a patch, and overlapping patches ensure comprehensive coverage. In this mode, it is necessary to input the model that will be used for patch-based inference in the subsequent steps.

Example of using:
```python
import cv2
from ultralytics import YOLO
from patched_yolo_infer import auto_calculate_crop_values

# Load the image
img_path = "test_image.jpg"
img = cv2.imread(img_path)

# Calculate the optimal crop size and overlap for an image
shape_x, shape_y, overlap_x, overlap_y = auto_calculate_crop_values(
    image=img, mode="network_based", model=YOLO("yolov8m.pt")
)
```