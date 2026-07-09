cmake_minimum_required(VERSION 3.16)
project(fusiontrack_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# --- Core static library ---------------------------------------------------
add_library(fusiontrack_core STATIC src/ekf.cpp)
target_include_directories(fusiontrack_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Position-independent code so the static core can be linked into the shared
# pybind11 module on Linux (macOS defaults to PIC; Linux does not).
set_target_properties(fusiontrack_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Unit tests (ctest) ----------------------------------------------------
enable_testing()
add_executable(test_ekf tests/test_ekf.cpp)
target_link_libraries(test_ekf PRIVATE fusiontrack_core)
add_test(NAME core_unit_tests COMMAND test_ekf)

# --- Optional Python bindings ---------------------------------------------
# Enable with: cmake -DFUSIONTRACK_PYBIND=ON .. (requires `pip install pybind11`)
option(FUSIONTRACK_PYBIND "Build the pybind11 Python module" OFF)
if(FUSIONTRACK_PYBIND)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(fusiontrack_cpp bindings/pymodule.cpp)
  target_link_libraries(fusiontrack_cpp PRIVATE fusiontrack_core)
endif()
