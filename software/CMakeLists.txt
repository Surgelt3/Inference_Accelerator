cmake_minimum_required(VERSION 3.20)
project(elec490 VERSION 0.0)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

if((${CMAKE_CXX_COMPILER_ID} STREQUAL GNU) OR (${CMAKE_CXX_COMPILER_ID} STREQUAL AppleClang) OR (${CMAKE_CXX_COMPILER_ID} STREQUAL Clang))
  message(STATUS "${CMAKE_CXX_COMPILER_ID} compiler detected, adding compile flags")
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
  endif()
  set(CMAKE_CXX_FLAGS "-Wall -Wno-sign-compare -Wno-unused-function -Wno-missing-braces")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-g -O3")

  
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
  message(STATUS "MSVC detected, adding compile flags")
  set(CMAKE_BUILD_TYPE Debug)
  # set(CMAKE_CXX_FLAGS "-Wall")
  set(CMAKE_CXX_FLAGS_DEBUG "/DEBUG /Zi")
  set(CMAKE_CXX_FLAGS_RELEASE "/Ox")

endif()

if(APPLE)
  include_directories("/usr/local/include") 
  link_directories("/usr/local/lib")
endif(APPLE)

# --------------------------------


# INCLUDE(FindProtobuf)
# find_package(Protobuf REQUIRED)
# include_directories(${ProtobufIncludePath})

add_executable(compiler)
set(compiler_sources importer.cpp neural_net.cpp compiler.cpp compiler_de.cpp)

include(FetchContent)
FetchContent_Declare(
  protobuf
  GIT_REPOSITORY	https://github.com/protocolbuffers/protobuf.git
  GIT_TAG 	v34.0
)
FetchContent_MakeAvailable(protobuf)

execute_process (
    COMMAND cp ${CMAKE_SOURCE_DIR}/onnx.pb.h ${CMAKE_CURRENT_BINARY_DIR}/
    OUTPUT_VARIABLE outVar
)

INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
# PROTOBUF_GENERATE_CPP(PROTO_SRC PROTO_HEADER onnx.proto)
set(PROTO_SRC onnx.pb.cc)
set(PROTO_HEADER onnx.pb.h)
ADD_LIBRARY(proto ${PROTO_HEADER} ${PROTO_SRC})
target_link_libraries(proto PUBLIC libprotobuf)

target_sources(compiler PRIVATE ${compiler_sources})
TARGET_LINK_LIBRARIES(compiler PRIVATE proto ${Protobuf_LIBRARIES})
target_include_directories(compiler PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(compiler PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(compiler PRIVATE ${PROTOBUF_INCLUDE_DIR})
target_include_directories(compiler PRIVATE "include/")

# -----------------------------------



