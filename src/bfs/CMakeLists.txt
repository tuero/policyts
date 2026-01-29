# Best First Search
add_executable(bfs_train bfs_train.cpp)
target_link_libraries(bfs_train libpolicyts::libpolicyts)
target_compile_features(bfs_train PUBLIC cxx_std_23)

add_executable(bfs_test bfs_test.cpp)
target_link_libraries(bfs_test libpolicyts::libpolicyts)
target_compile_features(bfs_test PUBLIC cxx_std_23)
