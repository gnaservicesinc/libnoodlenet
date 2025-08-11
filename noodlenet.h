// noodlenet.h
// NoodleNet C API Header File

#ifndef NOODLENET_H
#define NOODLENET_H

#include <stddef.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Activation function types supported by NoodleNet
 */
typedef enum {
    NN_ACTIVATION_FUNCTION_SIGMOID = 0,
    NN_ACTIVATION_FUNCTION_TANH = 1,     // New
    NN_ACTIVATION_FUNCTION_RELU = 2,     // New
    NN_ACTIVATION_FUNCTION_LEAKY_RELU = 3, // New
    // Keep this last for count
    NN_ACTIVATION_FUNCTION_COUNT
} ActivationFunction;

// Optimizers supported by training API
typedef enum {
    NN_OPTIMIZER_SGD = 0,
    NN_OPTIMIZER_RMSPROP = 1,
    NN_OPTIMIZER_ADAM = 2
} NN_Optimizer;

// Visualization options
typedef enum {
    NN_VIS_MODE_WEIGHTS = 0,   // Render weights as square image (resampled prev->curr)
    NN_VIS_MODE_HEATMAP = 1    // Cosine similarity heatmap between neuron weight vectors
} NN_VisMode;

typedef enum {
    NN_VIS_SCALE_MINMAX = 0,   // Map [min,max] -> [0,255]
    NN_VIS_SCALE_SYM_ZERO = 1  // Map [-max_abs,+max_abs] -> [0,255]
} NN_VisScale;

typedef struct {
    NN_VisMode mode;           // weights or heatmap
    NN_VisScale scale;         // min-max or symmetric zero-centered
    int include_bias;          // if nonzero, export biases as 1xN PGM per layer
    int include_stats;         // if nonzero, write stats text per layer
} NN_VisOptions;

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

// -------- Extended C API (Model + Training) --------

// Opaque model handle
typedef struct NN_Model NN_Model;

// Create a new model with the given architecture.
// hidden_sizes/hidden_activations have length num_hidden.
// Pass NULL for hidden_activations to default to SIGMOID.
NN_Model* nn_model_create(int input_neurons,
                          const int* hidden_sizes, int num_hidden,
                          int output_neurons,
                          const ActivationFunction* hidden_activations,
                          ActivationFunction output_activation);

// Load/Save model files (compatible with Sensuser .senm format)
NN_Model* nn_model_load(const char* model_path);
int       nn_model_save(const NN_Model* model, const char* model_path);
void      nn_model_free(NN_Model* model);

// Inspection helpers
int  nn_model_num_hidden(const NN_Model* model);
int  nn_model_hidden_size(const NN_Model* model, int layer_index); // 0-based
int  nn_model_input_size(const NN_Model* model);
int  nn_model_output_size(const NN_Model* model);
ActivationFunction nn_model_hidden_activation(const NN_Model* model, int layer_index);
ActivationFunction nn_model_output_activation(const NN_Model* model);

// Predict on an image file, returns 0 on success and writes probability [0,1] to out_prob
int nn_model_predict_image(const NN_Model* model, const char* image_path, float* out_prob);

// Simple SGD training configuration applied to directories of images
// Required: pos_dir; Optional: neg_dir, val_dir (can be NULL)
int nn_train_from_dirs(NN_Model* model,
                       const char* pos_dir,
                       const char* neg_dir,
                       const char* val_dir,
                       int steps,
                       int batch_size,
                       float learning_rate,
                       float l1_lambda,
                       float l2_lambda,
                       float* out_last_loss,
                       float* out_val_loss);

// Export a square grayscale PGM visualization per hidden layer (default: weights mode, min/max scaling)
int nn_export_layer_visualizations(const NN_Model* model, const char* output_dir);

// Extended export with options
int nn_export_layer_visualizations_ex(const NN_Model* model, const char* output_dir, const NN_VisOptions* options);

// Evaluate a model on directories of images.
// Returns 0 on success. Any of the output pointers may be NULL if not needed.
int nn_evaluate_dirs(const NN_Model* model,
                     const char* pos_dir,
                     const char* neg_dir,
                     int* true_pos,
                     int* true_neg,
                     int* false_pos,
                     int* false_neg,
                     float* out_accuracy);

// Optimizer configuration
int nn_model_set_optimizer(NN_Model* model, NN_Optimizer opt, float beta1, float beta2, float epsilon);

// Lightweight introspection API
int nn_num_weight_layers(const NN_Model* model);
int nn_layer_dims(const NN_Model* model, int layer_index, int* out_in, int* out_out);
// Copies weights (row-major: out x in) into caller buffer of length in*out
int nn_get_weights(const NN_Model* model, int layer_index, float* out, size_t out_len);
int nn_get_biases(const NN_Model* model, int layer_index, float* out, size_t out_len);
// Compute activations at a given layer for an image path; layer_index in [0..num_weight_sets], where index==num_weight_sets returns output layer activations
int nn_compute_activations_from_image(const NN_Model* model, const char* image_path, int layer_index, float* out, size_t out_len);
// Compute pre-activations z at a given layer for an image; valid layer_index in [1..num_weight_sets]
int nn_compute_pre_activations_from_image(const NN_Model* model, const char* image_path, int layer_index, float* out, size_t out_len);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NOODLENET_H
