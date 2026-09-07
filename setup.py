from setuptools import setup, find_packages

setup(
    name="yolo_object_detection",
    version="1.0.0",
    description="YOLOv8 Real-Time Traffic Cone Object Detection & Instance Counting Framework",
    author="Bhanu Vignesh Naidu Ganeshna",
    url="https://github.com/gbhanuvigneshnaidu29052002-droid/yolo-object-detection",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "pyyaml>=5.4.1",
        "pandas>=1.3.0"
    ],
    entry_points={
        'console_scripts': [
            'yolo-detect=main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires='>=3.8',
)
