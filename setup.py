from setuptools import setup, find_packages

setup(
    name="yolo_object_detection",
    version="1.0.0",
    description="YOLOv8 Traffic Cone Detection & Automated Counting Framework",
    author="Bhanu Vignesh Naidu Ganeshna",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0"
    ],
    entry_points={
        'console_scripts': [
            'yolo-detect=main:main',
        ],
    },
    python_requires='>=3.8',
)
