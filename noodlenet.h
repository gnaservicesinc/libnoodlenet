// noodlenet.h
// NoodleNet C API Header File

#ifndef NOODLENET_H
#define NOODLENET_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Predicts using a NoodleNet model.
 *
 * Loads a model specified by model_path, loads and processes the image
 * specified by image_path, and runs the image through the model.
 * The model file must be in the sensuser hybrid binary JSON format.
 * The image will be converted to 512x512 grayscale.
 *
 * @param model_path Path to the .model file.
 * @param image_path Path to the image file (e.g., PNG, JPG, BMP).
 * @return
 * 1 if the model predicts the object is present (output > 0.5).
 * 0 if the model predicts the object is not present (output <= 0.5).
 * -1 on any error (file access, model format, image format, memory allocation, etc.).
 */
int noodlenet_predict(const char* model_path, const char* image_path);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NOODLENET_H
