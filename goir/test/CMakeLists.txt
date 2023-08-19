configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(GO_TEST_DEPENDS
        FileCheck count not
        CGoIR-test
        go-opt
        #go-translate
)

add_lit_testsuite(check-go "Running the go regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${GO_TEST_DEPENDS}
)
set_target_properties(check-go PROPERTIES FOLDER "Tests")

add_lit_testsuites(Go ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${GO_TEST_DEPENDS})
add_subdirectory(CAPI)