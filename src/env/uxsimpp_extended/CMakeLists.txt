cmake_minimum_required(VERSION 3.15)
project(trafficppy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 使用 pybind11 创建 Python 绑定
find_package(pybind11 REQUIRED CONFIG)

# 创建 C++ 核心库
add_library(trafficppy_lib
    trafficpp/traffic.cpp
)

# 创建 Python 模块
pybind11_add_module(trafficppy
    trafficpp/bindings.cpp
)
target_link_libraries(trafficppy PRIVATE trafficppy_lib)

# 安装设置
install(TARGETS trafficppy
        LIBRARY DESTINATION uxsimpp_extended)

# 安装 Python 文件
install(FILES
    __init__.py
    uxsimpp.py
    DESTINATION uxsimpp_extended)

# 编译优化选项
if(MSVC)
    target_compile_options(trafficppy PRIVATE /O2)
else()
    target_compile_options(trafficppy PRIVATE -O3)
endif()

# Windows Mingw 环境静态链接
if (MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++ -static -O3")
endif()
