# Flex Transfer Engine library

# Collect source files
set(FLEX_TRANSFER_SOURCES
    copy_backend_rdma.cpp
    copy_backend_tcp.cpp
    copy_client.cpp
    copy_server.cpp
    flex_batch.cpp
    flex_transfer_engine.cpp
)

# Create the flex transfer engine library
add_library(flex_transfer_engine ${FLEX_TRANSFER_SOURCES})

# Include directories
target_include_directories(flex_transfer_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/mooncake-transfer-engine/include
)

# Link against transfer engine
target_link_libraries(flex_transfer_engine
    transfer_engine
)

# If CUDA is enabled, link CUDA runtime
if (USE_CUDA)
    target_link_libraries(flex_transfer_engine
        ${CUDA_LIBRARIES}
    )
endif()

# Install the library
install(TARGETS flex_transfer_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Build example program
if (BUILD_EXAMPLES)
    add_executable(flex_transfer_example example.cpp)
    target_link_libraries(flex_transfer_example
        flex_transfer_engine
        transfer_engine
    )
    install(TARGETS flex_transfer_example DESTINATION bin)
endif()
