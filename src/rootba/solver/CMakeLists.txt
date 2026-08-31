add_library(rootba_solver)

target_sources(rootba_solver
  PUBLIC
    bal_bundle_adjustment.hpp
    linearizor.hpp
    linearizor_base.hpp
    linearizor_qr.hpp
    solver_summary.hpp
  PRIVATE
    bal_bundle_adjustment.cpp
    linearizor.cpp
    linearizor_base.cpp
    linearizor_qr.cpp
)

target_link_libraries(rootba_solver
  PUBLIC
    rootba
    rootba_cg
    rootba_qr
)

if(ROOTBA_ENABLE_TESTING)

  rootba_add_test(test_librootba_solver
      bal_bundle_adjustment.test.cpp
    LINK_LIBRARIES
      rootba_solver
      rootba_testing
  )

endif()
