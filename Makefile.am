ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = libnoodlenet.la libnoodlenet_cpp.la

# C library
libnoodlenet_la_SOURCES = noodlenet.c cJSON.c
libnoodlenet_la_CFLAGS = -std=c11 -Wall -Wextra -O3
libnoodlenet_la_LDFLAGS = -version-info 0:0:0

# C++ library
libnoodlenet_cpp_la_SOURCES = noodlenet.cpp
libnoodlenet_cpp_la_CXXFLAGS = -std=c++11 -Wall -Wextra -O3
libnoodlenet_cpp_la_LDFLAGS = -version-info 0:0:0
libnoodlenet_cpp_la_LIBADD = libnoodlenet.la

# Install headers
include_HEADERS = noodlenet.h noodlenet.hpp cJSON.h stb_image.h

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libnoodlenet.pc

EXTRA_DIST = README.md
