# siirlts
add_executable(siirlts_train siirlts_train.cpp siirlts_rerooter.cpp)
target_link_libraries(siirlts_train libpolicyts::libpolicyts)
target_compile_features(siirlts_train PUBLIC cxx_std_23)

add_executable(siirlts_test siirlts_test.cpp siirlts_rerooter.cpp)
target_link_libraries(siirlts_test libpolicyts::libpolicyts)
target_compile_features(siirlts_test PUBLIC cxx_std_23)
