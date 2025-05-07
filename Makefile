# Makefile for NoodleNet Library

# Default C and C++ compilers
CC ?= gcc
CXX ?= g++
LD ?= $(CC) # Linker, usually the same as the C compiler for C projects or CXX for C++

# Common flags
CFLAGS_COMMON = -std=c11 -Wall -Wextra -O2
CXXFLAGS_COMMON = -std=c++11 -Wall -Wextra -O2
LDFLAGS_COMMON =

# Include directory (if headers are in a separate 'include' folder, not needed here)
# INCLUDES = -Iinclude

# Source files
C_SOURCES = noodlenet.c cJSON.c
CPP_SOURCES = noodlenet.cpp
OBJECTS_C = $(C_SOURCES:.c=.o)
OBJECTS_CPP = $(CPP_SOURCES:.cpp=.o)

# Library name
LIB_NAME = noodlenet

# Default target: build based on OS
# This is a simplistic OS detection. For robust builds, CMake or Autotools are better.
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Linux)
    TARGET_EXT = .so
    CFLAGS_OS = -fPIC
    CXXFLAGS_OS = -fPIC
    LDFLAGS_OS = -shared
    # For C++ library linking against C objects:
    # LDFLAGS_CPP_OS = -shared -Wl,-soname,lib$(LIB_NAME)$(TARGET_EXT)
endif
ifeq ($(UNAME_S),Darwin) # macOS
    TARGET_EXT = .dylib
    CFLAGS_OS = -fPIC
    CXXFLAGS_OS = -fPIC
    LDFLAGS_OS = -shared -dynamiclib
    # For C++ library linking against C objects:
    # LDFLAGS_CPP_OS = -shared -dynamiclib -install_name @rpath/lib$(LIB_NAME)$(TARGET_EXT)
endif
# For Windows (using MinGW or similar GCC-compatible environment)
# Check for OS variable, common in Windows build environments like MSYS2
ifeq ($(OS),Windows_NT)
    TARGET_EXT = .dll
    CFLAGS_OS =
    CXXFLAGS_OS =
    LDFLAGS_OS = -shared -Wl,--out-implib,lib$(LIB_NAME).a # Creates DLL and import library
    # For C++ library linking against C objects:
    # LDFLAGS_CPP_OS = -shared -Wl,--out-implib,lib$(LIB_NAME)_cpp.a
else ifneq (,$(findstring MINGW,$(UNAME_S))) # Heuristic for MinGW
    TARGET_EXT = .dll
    CFLAGS_OS =
    CXXFLAGS_OS =
    LDFLAGS_OS = -shared -Wl,--out-implib,lib$(LIB_NAME).a
endif

# Final flags
CFLAGS = $(CFLAGS_COMMON) $(CFLAGS_OS) $(INCLUDES)
CXXFLAGS = $(CXXFLAGS_COMMON) $(CXXFLAGS_OS) $(INCLUDES)
LDFLAGS = $(LDFLAGS_OS) $(LDFLAGS_COMMON)

# Targets
all: lib_c lib_cpp

lib_c: $(OBJECTS_C)
	$(LD) $(LDFLAGS) -o lib$(LIB_NAME)$(TARGET_EXT) $(OBJECTS_C)
	@echo "Built C Library: lib$(LIB_NAME)$(TARGET_EXT)"

lib_cpp: lib_c $(OBJECTS_CPP) # C++ library depends on C objects being available for linking
	$(CXX) $(LDFLAGS) -o lib$(LIB_NAME)_cpp$(TARGET_EXT) $(OBJECTS_CPP) $(OBJECTS_C) # Link C++ objects and C objects
	@echo "Built C++ Library: lib$(LIB_NAME)_cpp$(TARGET_EXT)"
	@echo "Note: The C++ library links against the C objects. Ensure lib_c is built or link C objects directly."


# Rule to compile C source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile C++ source files to object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS_C) $(OBJECTS_CPP) lib$(LIB_NAME)$(TARGET_EXT) lib$(LIB_NAME)_cpp$(TARGET_EXT) lib$(LIB_NAME).a
	@echo "Cleaned up build files."

.PHONY: all lib_c lib_cpp clean

# Example Usage:
# make           # Builds both C and C++ libraries for the detected OS
# make lib_c     # Builds only the C library
# make lib_cpp   # Builds only the C++ library (which also builds C objects)
# make clean     # Removes build artifacts
#
# To specify a compiler (e.g., clang):
# make CC=clang CXX=clang++ LD=clang
