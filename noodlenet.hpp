// noodlenet.hpp
// NoodleNet C++ API Header File

#ifndef NOODLENET_HPP
#define NOODLENET_HPP

#include <string>

// Forward declare the C function if not including noodlenet.h directly
// or ensure noodlenet.h is C++ compatible (which it is with extern "C").
#include "noodlenet.h" // Includes the C API

/**
 * @brief NoodleNet C++ Wrapper Class
 *
 * Provides a simple static method to perform predictions using the NoodleNet C library.
 */
class NoodleNet {
public:
    /**
     * @brief Predicts using a NoodleNet model.
     *
     * Loads a model specified by model_path, loads and processes the image
     * specified by image_path, and runs the image through the model.
     * This is a convenience wrapper around the C function noodlenet_predict.
     *
     * @param model_path Path to the .model file.
     * @param image_path Path to the image file (e.g., PNG, JPG, BMP).
     * @return
     * 1 if the model predicts the object is present.
     * 0 if the model predicts the object is not present.
     * -1 on any error.
     */
    static int predict(const std::string& model_path, const std::string& image_path) {
        return noodlenet_predict(model_path.c_str(), image_path.c_str());
    }
};

#endif // NOODLENET_HPP
