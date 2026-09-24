from setuptools import setup, find_packages

setup(
    name="locust_arena_calibration",
    version="0.0.1",
    description="Calibration utilities for locust arena (template)",
    author="Your Name",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "opencv-python",
        "PyYAML",
    ],
)
