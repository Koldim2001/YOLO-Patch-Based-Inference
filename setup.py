from setuptools import setup, find_packages
import codecs
import os

pwd = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(pwd, "patched_yolo_infer/README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


VERSION = '1.0.2'
DESCRIPTION = '''YOLO-Patch-Based-Inference for detection/segmentation of small objects in images.'''

setup(
    name="patched_yolo_infer",
    version=VERSION,
    license="MIT",
    url="https://github.com/Koldim2001/YOLO-Patch-Based-Inference",
    author="Koldim2001",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
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
        "rtdetr",
        "sam",
        "object detection",
        "instance segmentation",
        "patch-based inference",
        "small object detection",
        "yolov8-seg"
        "image patching"
        "yolo visualization"
        "slice-based inference",
        "slicing inference",
        "inference visualization",
        "patchify",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
