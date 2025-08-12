# libnoodlenet
A simple library for C and library for C++ to use models created with sensuser https://github.com/gnaservicesinc/sensuser in C and CPP projects.


This library has just one function you can call:

C API:
int noodlenet_predict(const char* model_path, const char* image_path);
C++ API:
static int NoodleNet::predict(const std::string& model_path, const std::string& image_path)

model_path should be a path to a .model file created with sensuser.
image_path should be a path to an image file (e.g., PNG, JPG, BMP).

Both functions return:

It will return 1 if the model predicts the object is present (output > 0.5).

It will return 0 if the model predicts the object is not present (output <= 0.5).

It will return -1 on any error (file access, model format, image format, memory allocation, etc.).

## Supported Model Features

The library supports models with the following features:
- Multiple hidden layers (new in version 2.0)
- Single hidden layer (backward compatible with version 1.0)
- Sigmoid activation function for all layers
- Binary classification (single output neuron)


Thats it. The library is designed to be easy to use with minimal API surface and no memory management required by the user


If you have linking issues when running make, try the alternative make file "NoodleNetMakefile"

make -f NoodleNetMakefile

## Performance and Build Options (macOS)

- Accelerated math: On macOS, the build links against the Apple Accelerate framework to speed up forward and training passes via BLAS. This is automatic when building on Darwin.
- Optimization flags: The C library is compiled with `-O3` by default.
- No GPU/MPS required: Current acceleration is CPU-based and leverages highly optimized vectorized routines.
