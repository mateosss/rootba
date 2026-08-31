add_library(rootba_qr)

target_sources(rootba_qr
  PUBLIC
    landmark_block.hpp
    linearization_qr.hpp
    linearization_utils.hpp
  PRIVATE
    landmark_block.cpp
    landmark_block_base.hpp
    landmark_block_dynamic.hpp
    impl/landmark_block_base.ipp
    impl/landmark_block_dynamic.cpp
)

target_link_libraries(rootba_qr
  PUBLIC
    rootba
)


if(ROOTBA_ENABLE_TESTING)

  rootba_add_test(test_librootba_qr
      linearization_qr.test.cpp
    LINK_LIBRARIES
      rootba_qr
      rootba_testing
  )

endif()
