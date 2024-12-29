from setuptools import setup, find_packages, Extension
import pybind11
import numpy as np

# C++ extension
ext_modules = [
    Extension(
        "gpgpu.core_ops",
        ["gpgpu/cpp/core_ops.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            np.get_include(),
        ],
        language='c++',
        extra_compile_args=['-std=c++14', '-O3'],
    ),
]

setup(
    name="gpgpu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyopencl>=2022.1",
        "numpy>=1.19.0",
        "numba>=0.56.0",
        "pybind11>=2.6.0",
    ],
    ext_modules=ext_modules,
    author="Your Name",
    author_email="your.email@example.com",
    description="A lightweight GPGPU module using PyOpenCL and C++ extensions",
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