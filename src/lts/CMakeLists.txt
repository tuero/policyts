# LevinTS
add_executable(lts_train lts_train.cpp)
target_link_libraries(lts_train libpolicyts::libpolicyts)
target_compile_features(lts_train PUBLIC cxx_std_23)

add_executable(lts_test lts_test.cpp)
target_link_libraries(lts_test libpolicyts::libpolicyts)
target_compile_features(lts_test PUBLIC cxx_std_23)
