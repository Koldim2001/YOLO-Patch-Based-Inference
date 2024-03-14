from setuptools import setup, find_packages
import codecs
import os

pwd = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(pwd, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


VERSION = '0.0.4'
DESCRIPTION = 'YOLO-Patch-Based-Detection'
LONG_DESCRIPTION = '''This library facilitates various visualizations of inference results from YOLOv8 and YOLOv9 models, 
    cropping with overlays, as well as a patch-based inference algorithm enabling detection of small objects in images. 
    It works for both object detection and instance segmentation tasks using YOLO models.'''

setup(
    name="patched_yolo_infer",
    version=VERSION,
    license="MIT",
    url="https://github.com/Koldim2001/YOLO-Patch-Based-Inference",
    author="Koldim2001",
    author_email="koldim2001@yandex.ru",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'torch',
        'ultralytics'
    ],
    keywords=[
        "python",
        "yolov8",
        "yolov9",
        "object detection",
        "instance segmentation",
        "patch-based inference",
        "small object detection",
        "yolov8-seg"
        "image patching"
        "yolo visualization"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
