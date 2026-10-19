cmake_minimum_required(VERSION 3.20)
project(Rover LANGUAGES C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
option(ROVER_HOST_TOOLS "Build workstation tools only" OFF)
option(ROVER_ENABLE_BUILD "Include graph construction" OFF)
if(ROVER_HOST_TOOLS)
  find_package(OpenMP)
  add_executable(rover-layout src/graph/layout_opt_kmeans.cpp)
  add_executable(rover-residual-layout tools/residual_layout.c)
  target_include_directories(rover-residual-layout PRIVATE src config)
  target_link_libraries(rover-residual-layout m)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(rover-layout PRIVATE OpenMP::OpenMP_CXX)
  endif()
  return()
endif()
if(NOT HEXAGON_SDK_ROOT)
  message(FATAL_ERROR "Set HEXAGON_SDK_ROOT to a locally installed SDK")
endif()
include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
set(NO_QURT_INC 1)
include_directories(src config ${CMAKE_CURRENT_BINARY_DIR}
  ${HEXAGON_SDK_ROOT}/incs ${HEXAGON_SDK_ROOT}/incs/stddef
  ${HEXAGON_SDK_ROOT}/libs/qhl_hvx/inc/qhblas_hvx)
if(OS_TYPE STREQUAL "HLOS")
  find_package(Threads REQUIRED)
  add_library(calculator SHARED ${CMAKE_CURRENT_BINARY_DIR}/calculator_stub.c)
  add_executable(Rover src/app/main.c src/app/config.c
    src/dsp_demo/retrieval_rpc_demo.c
    src/dsp_demo/retrieval_nsg_loader.c
    src/dsp_demo/retrieval_sqmd_seed.c
    src/dsp_demo/retrieval_rnn_descent_core.c
    src/dsp_demo/retrieval_search_build_core.c
    src/dsp_demo/retrieval_raw_reader.c
    src/dsp_demo/retrieval_final_refine.c
    src/dsp_demo/retrieval_build_artifacts.c
    src/dsp_demo/retrieval_pq_core.c
    src/dsp_demo/retrieval_pq_artifact.c
    src/dsp_demo/retrieval_pq_residual.c
    src/dsp_demo/retrieval_sqmd_pure_pq.c
    src/dsp_demo/retrieval_pure_pq_index.c
    src/dsp_demo/retrieval_pure_pq_builder.c
    src/imp/quantize_compress.c
    src/dsp_demo/retrieval_l1_int8_core.c)
  if(ROVER_ENABLE_BUILD)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/build/rover_insert_build_core.c")
      message(FATAL_ERROR "Checkout graph-build-insert to compile construction")
    endif()
    target_sources(Rover PRIVATE src/build/rover_insert_build_core.c src/dsp_demo/retrieval_insert_stream.c)
    target_compile_definitions(Rover PRIVATE ROVER_ENABLE_BUILD=1 RETRIEVAL_ENABLE_FULL_BUILD=1)
  endif()
  target_compile_definitions(Rover PRIVATE VERIFY_PRINT_ERROR)
  set_common_compile_and_link_options(Rover)
  build_idl(inc/calculator.idl calculator)
  choose_dsprpc(${DSP_TYPE} dsprpc)
  link_custom_library(Rover rpcmem)
  link_custom_library(calculator ${dsprpc})
  link_custom_library(Rover ${dsprpc})
  target_link_libraries(Rover calculator m dl Threads::Threads)
  add_dependencies(Rover calculator)
else()
  add_library(calculator_skel_OBJS OBJECT
    ${CMAKE_CURRENT_BINARY_DIR}/calculator_skel.c src/app/dsp.c
    src/dsp_demo/retrieval_pq_core.c)
  target_compile_options(calculator_skel_OBJS PRIVATE -mhvx -mhvx-length=128B)
  if(ROVER_ENABLE_BUILD)
    target_compile_definitions(calculator_skel_OBJS PRIVATE RETRIEVAL_ENABLE_FULL_BUILD=1)
  endif()
  add_library(calculator_skel SHARED $<TARGET_OBJECTS:calculator_skel_OBJS>)
  link_custom_library(calculator_skel qhl_hvx)
  build_idl(inc/calculator.idl calculator_skel_OBJS)
endif()
