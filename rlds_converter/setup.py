from setuptools import setup

setup(
    name="dlimp-dataset-builder",
    python_requires=">=3.10",
    install_requires=[
        "tensorflow_datasets==4.9.4",
        "opencv-python",
        "apache_beam",
        "tensorflow",
    ],
)
