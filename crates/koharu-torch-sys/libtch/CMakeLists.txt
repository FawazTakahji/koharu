cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(tch CXX)

# The LibTorch installation root is resolved directly instead of through
# find_package(Torch), so the shim build does not depend on the CMake package
# configs being present in every distribution. KOHARU_TORCH_ROOT must point at
# a LibTorch installation with include/ and lib/.
set(KOHARU_TORCH_ROOT "" CACHE PATH "Root of a LibTorch installation")

add_library(koharu-torch SHARED torch_api.cpp torch_api_generated.cpp)
target_include_directories(koharu-torch PRIVATE
    "${KOHARU_TORCH_ROOT}/include"
    "${KOHARU_TORCH_ROOT}/include/torch/csrc/api/include"
)
target_link_directories(koharu-torch PRIVATE "${KOHARU_TORCH_ROOT}/lib")
target_link_libraries(koharu-torch torch torch_cpu c10)
if(MSVC)
    # Mirrors the interface compile options exported by LibTorch's CMake
    # targets, which find_package would otherwise provide.
    target_compile_options(koharu-torch PRIVATE /permissive- /EHsc /bigobj)
endif()
set_property(TARGET koharu-torch PROPERTY CXX_STANDARD 20)
set_property(TARGET koharu-torch PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
install(TARGETS koharu-torch DESTINATION .)
