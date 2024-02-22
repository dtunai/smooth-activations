cmake_minimum_required(VERSION 3.4)

# set project
project(smelu)

# set standart
set(CMAKE_CXX_STANDARD 11)

# enable CUDA language support for linker
if(CUDA_ENABLE)
    enable_language(CUDA)
endif()

# set dependency vars
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0")
set(PYBIND11_INSTALL_DIR "C:/Users/<USER>/AppData/Local/Programs/Python/Python310/Lib/site-packages/pybind11")
set(CUDA_NVCC_EXECUTABLE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")

# find pybind11 and CUDA packages
find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_INSTALL_DIR}/share/cmake/pybind11")

# set nvcc compiler as the CMAKE CUDA compiler
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")

# find CUDA package
find_package(CUDA REQUIRED)

# define the SmeLU module
add_library(smelu_module SHARED
        _csrc/smelu_kernels/smelu.h
        _csrc/smelu_kernels/smelu_kernels.cu
        _csrc/smelu_kernels/pybind_smelu.cpp)

list(APPEND SOURCES smelu_kernels.cu)

# set the output name to smelu
set_target_properties(smelu_module PROPERTIES OUTPUT_NAME smelu SUFFIX ".pyd")

# link the module to pybind11
target_link_libraries(smelu_module PRIVATE pybind11::module)

# set CUDA-specific annotation flags
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=/wd4819")
target_compile_definitions(smelu_module PRIVATE USE_CUDA)
target_include_directories(smelu_module PRIVATE ${CUDA_INCLUDE_DIRS})
target_compile_options(smelu_module PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# move the output to the smelu_kernel dir
set_target_properties(smelu_module PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/smelu_kernel"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/smelu_kernel"
        )

# use nvcc compiler to compile CUDA source files
set_source_files_properties(_csrc/smelu_kernels/smelu_kernels.cu PROPERTIES LANGUAGE CUDA)

# set nvcc flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -Xcompiler=-fPIC -lineinfo")

# set CUDA architecture flags
set(CMAKE_CUDA_ARCHITECTURES "70")

# link CUDA libraries
target_link_libraries(smelu_module PRIVATE ${CUDA_LIBRARIES})
