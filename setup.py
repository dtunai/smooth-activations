import os
import platform
import subprocess
import sys
import setuptools
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    """CMake extension class."""
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """CMake build command class."""
    def run(self):
        try:
            subprocess.check_call(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [ '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable ]

        build_args = []
        if self.debug:
            cmake_args += [ '-DCMAKE_BUILD_TYPE=Debug' ]
            build_args = [ '--config', 'Debug' ]
        else:
            cmake_args += [ '-DCMAKE_BUILD_TYPE=Release' ]
            build_args = [ '--config', 'Release' ]

        cmake_args += [ '-DCMAKE_CUDA_FLAGS=-arch=sm_70' ]

        if platform.system() == "Windows":
            cuda_compiler_path = os.path.join('C:', 'Program Files', 'NVIDIA GPU Computing Toolkit', 'CUDA', 'v12.0', 'bin', 'nvcc.exe')
            cmake_args += [ '-DCMAKE_CUDA_COMPILER=' + cuda_compiler_path ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='smelu-cuda',
    version='0.0.1',
    author='simudt',
    author_email='dogukanuraztuna@gmail.com',
    description='SmeLU (Smooth ReLU activations) with CUDA Kernel',
    url='https://github.com/simudt/smelu-cuda',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Windows :: 10',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.18.1',
    ],
    ext_modules=[CMakeExtension('smelu', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
