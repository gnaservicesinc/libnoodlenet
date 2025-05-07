// noodlenet.cpp
// NoodleNet C++ API Implementation

#include "noodlenet.hpp"

// The C++ NoodleNet::predict function is fully implemented in noodlenet.hpp
// as it's a simple static inline wrapper around the C function.
// This .cpp file is here to allow the Makefile to build it as part of a
// C++ library target if desired, or if more complex C++ specific logic
// were to be added later. For now, it can be empty or contain non-inline
// C++ specific implementations if NoodleNet class grew.

// If you wanted to ensure there's an object file generated for a C++ library,
// you could have a dummy function or instantiate something, but for the current
// simple wrapper, it's not strictly necessary if the header is included and used.
// However, makefiles often expect a .cpp for a .o target.

// Example of a non-inline static member definition if needed:
// (Not needed for the current NoodleNet::predict)
/*
int NoodleNet::some_other_static_method() {
    return 42;
}
*/
