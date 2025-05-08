# Bypass "ADD_LIBRARY called with SHARED option" error on some systems with an older CMake version.
set(CMAKE_SYSTEM_NAME Linux)
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE CACHE)