cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(torch_cluster_operators LANGUAGES CXX CUDA)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Define our library target
set(HEADERS
init.h
)

set(SOURCES
fps.cpp
init.cpp
knn.cpp
)

set(CUDA_SOURCES
atomics.cuh
compat.cuh
fps_kernel.cu
knn_kernel.cu
utils.cuh
)

add_library(torch_cluster_operators SHARED ${HEADERS} ${SOURCES} ${CUDA_SOURCES})

# Custom flags
if(MSVC)
    set(CMAKE_CUDA_FLAGS "-Xcompiler=\"/EHsc /wd4267 /wd4251 /wd4522 /wd4838 /wd4305 /wd4244 /wd4190 /wd4101 /wd4996 /wd4275 /bigobj\"")
endif()

# Enable C++14
target_compile_features(torch_cluster_operators PRIVATE cxx_range_for)
target_compile_features(torch_cluster_operators PUBLIC cxx_std_14)

# Link against LibTorch
target_link_libraries(torch_cluster_operators torch cuda)

if(${CMAKE_VERSION} GREATER_EQUAL "3.18")
    set_property(TARGET torch_cluster_operators PROPERTY CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86)
endif()

if (MSVC)
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        install(TARGETS torch_cluster_operators EXPORT torch_cluster_operators
            ARCHIVE DESTINATION lib/${OUTPUTCONFIG}
            LIBRARY DESTINATION lib/${OUTPUTCONFIG}
            RUNTIME DESTINATION bin/${OUTPUTCONFIG})

        install(FILES $<TARGET_PDB_FILE:torch_cluster_operators> DESTINATION lib/${OUTPUTCONFIG} OPTIONAL)

        install(EXPORT torch_cluster_operators DESTINATION lib/cmake/torch_cluster_operators FILE torch_cluster_operators-config.cmake)
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
else()
    install(TARGETS torch_cluster_operators EXPORT torch_cluster_operators
        ARCHIVE DESTINATION lib/torch_cluster_operators
        LIBRARY DESTINATION lib/torch_cluster_operators
        RUNTIME DESTINATION bin/torch_cluster_operators)
    install(EXPORT torch_cluster_operators DESTINATION lib/cmake/torch_cluster_operators FILE torch_cluster_operators-config.cmake)
endif(MSVC)
