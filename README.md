# libnoodlenet
A simple libary for C and libary for C++ to use models created with sensuser https://github.com/gnaservicesinc/sensuser in C and CPP projects.


This libary has just one function you can call: 

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


Thats it. The library is designed to be easy to use with minimal API surface and no memory management required by the user


If you have linking issues when running make, try the alternative make file "NoodleNetMakefile"

make -f NoodleNetMakefile
