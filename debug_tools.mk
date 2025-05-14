CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -g
CXXFLAGS = -Wall -Wextra -g
LDFLAGS = -lm

# Qt settings
QT_CXXFLAGS = $(shell pkg-config --cflags Qt5Core Qt5Gui Qt5Widgets)
QT_LDFLAGS = $(shell pkg-config --libs Qt5Core Qt5Gui Qt5Widgets)

# Paths
SENSUSER_PATH = ../sensuser
LIBNOODLENET_PATH = .
LIBNOODLENET_INCLUDE = -I$(LIBNOODLENET_PATH)
SENSUSER_INCLUDE = -I$(SENSUSER_PATH)

# Debug programs
all: debug_model_prediction debug_noodlenet noodlenet_bilinear.o

# Debug program for comparing sensuser and libnoodlenet
debug_model_prediction: debug_model_prediction.cpp
	$(CXX) $(CXXFLAGS) $(QT_CXXFLAGS) $(SENSUSER_INCLUDE) $(LIBNOODLENET_INCLUDE) -o $@ $< $(QT_LDFLAGS) -lnoodlenet

# Debug program for libnoodlenet
debug_noodlenet: debug_noodlenet.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lcjson

# Bilinear interpolation version of noodlenet
noodlenet_bilinear.o: noodlenet_bilinear.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Create a shared library with the bilinear version
libnoodlenet_bilinear.so: noodlenet_bilinear.o
	$(CC) -shared -o $@ $< -lm

clean:
	rm -f debug_model_prediction debug_noodlenet *.o *.so

.PHONY: all clean
