from setuptools import setup, find_packages

setup(
    name="gpgpu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyopencl>=2022.1",
        "numpy>=1.19.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A lightweight GPGPU module using PyOpenCL",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gpgpu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)