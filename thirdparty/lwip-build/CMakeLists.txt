cmake_minimum_required(VERSION 3.10)
project(lwip-sysroot C)

set(LWIP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../lwip)
include(${LWIP_DIR}/src/Filelists.cmake)

add_library(lwip STATIC ${lwipnoapps_SRCS})
target_include_directories(lwip
    PUBLIC  ${LWIP_DIR}/src/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}   # lwipopts.h
            ${CMAKE_SYSROOT}/include      # picolibc headers
)

install(TARGETS lwip ARCHIVE DESTINATION lib)
install(DIRECTORY ${LWIP_DIR}/src/include/lwip  DESTINATION include)
install(DIRECTORY ${LWIP_DIR}/src/include/netif DESTINATION include)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/arch DESTINATION include)
