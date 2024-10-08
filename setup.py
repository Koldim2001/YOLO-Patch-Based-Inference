from setuptools import setup, find_packages
import codecs
import os

pwd = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(pwd, "patched_yolo_infer/README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


VERSION = '1.3.3'
DESCRIPTION = '''Patch-Based-Inference for detection/segmentation of small objects in images.'''

setup(
    name="patched_yolo_infer",
    version=VERSION,
    license="AGPL-3.0 license",
    url="https://github.com/Koldim2001/YOLO-Patch-Based-Inference",
    author="Koldim2001",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'numpy<2.0',
        'opencv-python',
        'matplotlib',
        'ultralytics'
    ],
    keywords=[
        "python",
        "yolov8",
        "yolov9",
        "yolov10",
        "yolov11",
        "rtdetr",
        "fastsam",
        "sahi",
        "object detection",
        "instance segmentation",
        "patch-based inference",
        "small object detection",
        "yolov8-seg",
        "image patching",
        "yolo visualization",
        "slice-based inference",
        "slicing inference",
        "inference visualization",
        "patchify",
        "ultralytics",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
    ],
)
