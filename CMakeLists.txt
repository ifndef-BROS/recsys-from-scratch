cmake_minimum_required(VERSION 3.10)
project(Recommender)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(
    main
    src/main.cpp
    src/utils/data_loader.cpp
    src/retrieval/bloom_filter.cpp
    src/retrieval/kdtree.cpp 
    src/embeddings/user_embedding.cpp
    src/ranking/ranker.cpp
    src/pipeline.cpp
)

target_compile_options(main PRIVATE -O3 -march=native -mavx2 -mfma)
target_include_directories(main PRIVATE include)
