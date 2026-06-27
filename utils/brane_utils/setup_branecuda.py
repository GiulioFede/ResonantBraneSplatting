from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Change this path depending on where you place your C++/CUDA files
file_path = "utils/brane_cuda"

setup(
    name="branecuda", 
    ext_modules=[
        CUDAExtension(
            name="branecuda", 
            sources=[
                os.path.join(file_path, "branewrapper.cpp"),
                os.path.join(file_path, "brane.cu")
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)