# PHS*
add_executable(phs_train phs_train.cpp)
target_link_libraries(phs_train libpolicyts::libpolicyts)
target_compile_features(phs_train PUBLIC cxx_std_23)

add_executable(phs_test phs_test.cpp)
target_link_libraries(phs_test libpolicyts::libpolicyts)
target_compile_features(phs_test PUBLIC cxx_std_23)
