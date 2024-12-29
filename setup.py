from setuptools import setup, find_packages, Extension
import pybind11
import numpy as np

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
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "pyopencl>=2022.1",
        "numpy>=1.19.0",
        "numba>=0.56.0",
        "pybind11>=2.6.0",
    ],
    ext_modules=ext_modules,
    author="OpenGPGPU Team",
    author_email="contact@rndmcoolawsmgrbg.lol",
    description="A lightweight GPGPU module using PyOpenCL and C++ extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/open-gpgpu/gpgpu",
    python_requires=">=3.7",
)