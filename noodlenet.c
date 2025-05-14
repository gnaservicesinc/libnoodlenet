// noodlenet.c
// NoodleNet C API Implementation

#include "noodlenet.h"
#include "cJSON.h" // For parsing JSON metadata
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // For expf (sigmoid)

// STB Image Loading Library (Single Header)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Original nearest-neighbor resize function (kept for reference)
static int simple_resize_image(const unsigned char* src, int src_width, int src_height, int src_channels,
                              unsigned char* dst, int dst_width, int dst_height) {
    // Simple nearest-neighbor resize
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;

    for (int y = 0; y < dst_height; y++) {
        int src_y = (int)(y * y_ratio);
        if (src_y >= src_height) src_y = src_height - 1;

        for (int x = 0; x < dst_width; x++) {
            int src_x = (int)(x * x_ratio);
            if (src_x >= src_width) src_x = src_width - 1;

            // Copy all channels
            for (int c = 0; c < src_channels; c++) {
                dst[(y * dst_width + x) * src_channels + c] =
                    src[(src_y * src_width + src_x) * src_channels + c];
            }
        }
    }

    return 1; // Success
}

// Improved bilinear resize function for better quality
static int bilinear_resize_image(const unsigned char* src, int src_width, int src_height, int src_channels,
                              unsigned char* dst, int dst_width, int dst_height) {
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;

    for (int y = 0; y < dst_height; y++) {
        float src_y = y * y_ratio;
        int src_y_floor = (int)src_y;
        int src_y_ceil = (src_y_floor == src_height - 1) ? src_y_floor : src_y_floor + 1;
        float y_diff = src_y - src_y_floor;

        for (int x = 0; x < dst_width; x++) {
            float src_x = x * x_ratio;
            int src_x_floor = (int)src_x;
            int src_x_ceil = (src_x_floor == src_width - 1) ? src_x_floor : src_x_floor + 1;
            float x_diff = src_x - src_x_floor;

            for (int c = 0; c < src_channels; c++) {
                // Get the four surrounding pixels
                unsigned char top_left = src[(src_y_floor * src_width + src_x_floor) * src_channels + c];
                unsigned char top_right = src[(src_y_floor * src_width + src_x_ceil) * src_channels + c];
                unsigned char bottom_left = src[(src_y_ceil * src_width + src_x_floor) * src_channels + c];
                unsigned char bottom_right = src[(src_y_ceil * src_width + src_x_ceil) * src_channels + c];

                // Bilinear interpolation
                float top = top_left * (1 - x_diff) + top_right * x_diff;
                float bottom = bottom_left * (1 - x_diff) + bottom_right * x_diff;
                float pixel = top * (1 - y_diff) + bottom * y_diff;

                dst[(y * dst_width + x) * src_channels + c] = (unsigned char)pixel;
            }
        }
    }
    return 1;
}

// --- Configuration & Constants ---
#define TARGET_WIDTH 512
#define TARGET_HEIGHT 512
#define EXPECTED_INPUT_NEURONS (TARGET_WIDTH * TARGET_HEIGHT)
#define EXPECTED_OUTPUT_NEURONS 1
#define PREDICTION_THRESHOLD 0.5f
#define MODEL_MAGIC_NUMBER 0x4D4E4553 // "SENM" in little-endian (S E N M)
#define MODEL_MAGIC_NUMBER_REVERSED 0x53454E4D // "SENM" in big-endian (M N E S)
#define MODEL_FORMAT_VERSION 2

// --- Helper Structures ---
typedef struct {
    int input_neurons;
    int num_hidden_layers;
    int* hidden_neurons_per_layer; // Array of sizes for each hidden layer
    int output_neurons;
    // For simplicity, we assume "sigmoid" activation and "float" precision
    // These would be parsed and validated from JSON
} ModelArchitecture;

typedef struct {
    ModelArchitecture arch;
    float** weights; // Array of weight matrices (weights[0] = input_to_hidden1, weights[1] = hidden1_to_hidden2, ...)
    float** biases;  // Array of bias vectors
    int num_weight_sets; // Equal to num_hidden_layers + 1
} MLPModel;

// --- Forward Declarations of Static Helper Functions ---
static void free_mlp_model(MLPModel* model);
static float sigmoid(float x);
static int simple_resize_image(const unsigned char* src, int src_width, int src_height, int src_channels,
                              unsigned char* dst, int dst_width, int dst_height);
static int bilinear_resize_image(const unsigned char* src, int src_width, int src_height, int src_channels,
                              unsigned char* dst, int dst_width, int dst_height);
static int load_model_from_file(const char* model_path, MLPModel* model);
static int load_and_process_image(const char* image_path, float* output_buffer);
static float perform_forward_pass(const MLPModel* model, const float* input_data);

// --- Public API Implementation ---
int noodlenet_predict(const char* model_path, const char* image_path) {
    MLPModel model = {0};
    float* image_data = NULL;
    int result = -1; // Default to error

    // 1. Load the model
    if (load_model_from_file(model_path, &model) != 0) {
        fprintf(stderr, "NoodleNet Error: Failed to load model from '%s'\n", model_path);
        goto cleanup; // Error already printed in load_model_from_file
    }

    // 2. Load and process the image
    // Allocate buffer for the flattened, normalized grayscale image data
    image_data = (float*)malloc(EXPECTED_INPUT_NEURONS * sizeof(float));
    if (!image_data) {
        fprintf(stderr, "NoodleNet Error: Failed to allocate memory for image data.\n");
        goto cleanup;
    }
    if (load_and_process_image(image_path, image_data) != 0) {
        fprintf(stderr, "NoodleNet Error: Failed to load or process image '%s'\n", image_path);
        goto cleanup;
    }

    // 3. Perform forward pass
    float prediction = perform_forward_pass(&model, image_data);
    if (isnan(prediction) || isinf(prediction)) { // Check for invalid float results
        fprintf(stderr, "NoodleNet Error: Invalid prediction value (NaN or Inf).\n");
        goto cleanup;
    }

    // 4. Determine result based on threshold
    if (prediction > PREDICTION_THRESHOLD) {
        result = 1; // Object present
    } else {
        result = 0; // Object not present
    }

cleanup:
    if (image_data) {
        free(image_data);
    }
    free_mlp_model(&model); // Safe to call even if parts are NULL
    return result;
}

// --- Static Helper Function Implementations ---

static void free_mlp_model(MLPModel* model) {
    if (!model) return;

    if (model->arch.hidden_neurons_per_layer) {
        free(model->arch.hidden_neurons_per_layer);
        model->arch.hidden_neurons_per_layer = NULL;
    }

    if (model->weights) {
        for (int i = 0; i < model->num_weight_sets; ++i) {
            if (model->weights[i]) {
                free(model->weights[i]);
            }
        }
        free(model->weights);
        model->weights = NULL;
    }

    if (model->biases) {
        // Biases correspond to hidden layers + output layer
        for (int i = 0; i < model->arch.num_hidden_layers + 1; ++i) {
            if (model->biases[i]) {
                free(model->biases[i]);
            }
        }
        free(model->biases);
        model->biases = NULL;
    }
    model->num_weight_sets = 0;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static int load_model_from_file(const char* model_path, MLPModel* model) {
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        fprintf(stderr, "NoodleNet Error: Cannot open model file '%s'.\n", model_path);
        return -1;
    }

    int error_occurred = 0;
    char* json_str = NULL;
    cJSON* root_json = NULL;

    // Read Magic Number
    unsigned int magic_num_read;
    if (fread(&magic_num_read, sizeof(unsigned int), 1, fp) != 1) { error_occurred = 1; goto read_error; }
    // Check for both possible magic number formats for compatibility
    if (magic_num_read != MODEL_MAGIC_NUMBER && magic_num_read != MODEL_MAGIC_NUMBER_REVERSED) {
        fprintf(stderr, "NoodleNet Error: Invalid model magic number. Expected 0x%X or 0x%X, got 0x%X\n",
                MODEL_MAGIC_NUMBER, MODEL_MAGIC_NUMBER_REVERSED, magic_num_read);
        error_occurred = 1; goto read_error;
    }

    // Read Format Version
    unsigned char version_read;
    if (fread(&version_read, sizeof(unsigned char), 1, fp) != 1) { error_occurred = 1; goto read_error; }
    if (version_read != MODEL_FORMAT_VERSION && version_read != 1) {
        fprintf(stderr, "NoodleNet Error: Unsupported model format version. Expected %d or 1, got %d\n", MODEL_FORMAT_VERSION, version_read);
        error_occurred = 1; goto read_error;
    }

    // Read JSON Metadata Length
    unsigned int json_len;
    if (fread(&json_len, sizeof(unsigned int), 1, fp) != 1) { error_occurred = 1; goto read_error; }
    if (json_len == 0 || json_len > 1024 * 1024) { // Sanity check for JSON length (e.g., max 1MB)
         fprintf(stderr, "NoodleNet Error: Invalid JSON metadata length: %u\n", json_len);
         error_occurred = 1; goto read_error;
    }

    // Read JSON Metadata Block
    json_str = (char*)malloc(json_len + 1);
    if (!json_str) { fprintf(stderr, "NoodleNet Error: Malloc failed for JSON string.\n"); error_occurred = 1; goto read_error; }
    if (fread(json_str, 1, json_len, fp) != json_len) { error_occurred = 1; goto read_error; }
    json_str[json_len] = '\0';

    // Parse JSON
    root_json = cJSON_Parse(json_str);
    if (!root_json) {
        fprintf(stderr, "NoodleNet Error: Failed to parse JSON metadata. Error near: %s\n", cJSON_GetErrorPtr());
        error_occurred = 1; goto parse_error;
    }

    cJSON* arch_json = cJSON_GetObjectItemCaseSensitive(root_json, "architecture");
    if (!cJSON_IsObject(arch_json)) { error_occurred = 1; goto parse_error_msg; }

    cJSON* precision_json = cJSON_GetObjectItemCaseSensitive(root_json, "data_precision");
    if (!cJSON_IsString(precision_json) || strcmp(precision_json->valuestring, "float") != 0) {
        fprintf(stderr, "NoodleNet Error: Model data_precision must be 'float'.\n");
        error_occurred = 1; goto parse_error_msg;
    }

    // --- Populate ModelArchitecture ---
    model->arch.input_neurons = cJSON_GetObjectItemCaseSensitive(arch_json, "input_neurons")->valueint;
    model->arch.output_neurons = cJSON_GetObjectItemCaseSensitive(arch_json, "output_neurons")->valueint;

    if (model->arch.input_neurons != EXPECTED_INPUT_NEURONS || model->arch.output_neurons != EXPECTED_OUTPUT_NEURONS) {
        fprintf(stderr, "NoodleNet Error: Model architecture mismatch (input/output neurons).\n");
        error_occurred = 1; goto parse_error_msg;
    }

    // For simplicity, assuming "sigmoid" for all activations
    cJSON* output_act_json = cJSON_GetObjectItemCaseSensitive(arch_json, "output_activation");
    if (!cJSON_IsString(output_act_json) || strcmp(output_act_json->valuestring, "sigmoid") != 0) {
        fprintf(stderr, "NoodleNet Error: Model output activation function must be 'sigmoid' for this library version.\n");
        error_occurred = 1; goto parse_error_msg;
    }

    // Get hidden layers configuration
    cJSON* hidden_layers_json = cJSON_GetObjectItemCaseSensitive(arch_json, "hidden_layers");
    if (!cJSON_IsArray(hidden_layers_json)) {
        // Try old format key for backward compatibility
        hidden_layers_json = cJSON_GetObjectItemCaseSensitive(arch_json, "hidden_layers_neurons");
        if (!cJSON_IsArray(hidden_layers_json)) {
            error_occurred = 1; goto parse_error_msg;
        }
    }

    model->arch.num_hidden_layers = cJSON_GetArraySize(hidden_layers_json);

    // Allow models with no hidden layers (direct input to output) in format version 2
    if (version_read == 1 && model->arch.num_hidden_layers <= 0) {
        fprintf(stderr, "NoodleNet Error: Format version 1 models must have at least one hidden layer.\n");
        error_occurred = 1; goto parse_error_msg;
    }

    model->arch.hidden_neurons_per_layer = (int*)malloc(model->arch.num_hidden_layers * sizeof(int));
    if (!model->arch.hidden_neurons_per_layer && model->arch.num_hidden_layers > 0) {
        error_occurred = 1; goto malloc_fail_msg;
    }

    for (int i = 0; i < model->arch.num_hidden_layers; ++i) {
        cJSON* layer_item = cJSON_GetArrayItem(hidden_layers_json, i);
        int neurons = 0;

        if (version_read == 2) {
            // New format: each item is an object with "neurons" and "activation"
            if (!cJSON_IsObject(layer_item)) { error_occurred = 1; goto parse_error_msg; }

            cJSON* neurons_json = cJSON_GetObjectItemCaseSensitive(layer_item, "neurons");
            if (!cJSON_IsNumber(neurons_json) || neurons_json->valueint <= 0) {
                error_occurred = 1; goto parse_error_msg;
            }
            neurons = neurons_json->valueint;

            // Check activation (must be sigmoid for this library version)
            cJSON* act_json = cJSON_GetObjectItemCaseSensitive(layer_item, "activation");
            if (!cJSON_IsString(act_json) || strcmp(act_json->valuestring, "sigmoid") != 0) {
                fprintf(stderr, "NoodleNet Error: Model hidden layer activation functions must be 'sigmoid' for this library version.\n");
                error_occurred = 1; goto parse_error_msg;
            }
        } else {
            // Old format: each item is just the number of neurons
            if (!cJSON_IsNumber(layer_item) || layer_item->valueint <= 0) {
                error_occurred = 1; goto parse_error_msg;
            }
            neurons = layer_item->valueint;
        }

        model->arch.hidden_neurons_per_layer[i] = neurons;
    }

    // --- Allocate memory for weights and biases ---
    model->num_weight_sets = model->arch.num_hidden_layers + 1;
    model->weights = (float**)calloc(model->num_weight_sets, sizeof(float*));
    model->biases = (float**)calloc(model->arch.num_hidden_layers + 1, sizeof(float*)); // Biases for hidden layers + output layer
    if (!model->weights || !model->biases) { error_occurred = 1; goto malloc_fail_msg; }

    // Sizes for iteration
    int prev_layer_size = model->arch.input_neurons;
    long total_floats_to_read = 0;

    // Weights and biases for input -> hidden1, hidden_i -> hidden_{i+1}, ..., last_hidden -> output
    for (int i = 0; i < model->num_weight_sets; ++i) {
        int current_layer_size;
        if (i < model->arch.num_hidden_layers) { // It's a hidden layer
            current_layer_size = model->arch.hidden_neurons_per_layer[i];
        } else { // It's the output layer
            current_layer_size = model->arch.output_neurons;
        }

        long num_weights_in_set = (long)prev_layer_size * current_layer_size;
        model->weights[i] = (float*)malloc(num_weights_in_set * sizeof(float));
        model->biases[i] = (float*)malloc(current_layer_size * sizeof(float)); // Biases for current layer

        if (!model->weights[i] || !model->biases[i]) { error_occurred = 1; goto malloc_fail_msg; }

        total_floats_to_read += num_weights_in_set;
        total_floats_to_read += current_layer_size; // for biases

        prev_layer_size = current_layer_size;
    }

    // --- Read Binary Data Block (Weights and Biases) ---
    // Reset prev_layer_size for reading loop
    prev_layer_size = model->arch.input_neurons;
    for (int i = 0; i < model->num_weight_sets; ++i) {
        int current_layer_size;
        if (i < model->arch.num_hidden_layers) {
            current_layer_size = model->arch.hidden_neurons_per_layer[i];
        } else {
            current_layer_size = model->arch.output_neurons;
        }

        long num_weights_to_read = (long)prev_layer_size * current_layer_size;
        if (fread(model->weights[i], sizeof(float), num_weights_to_read, fp) != (size_t)num_weights_to_read) {
            error_occurred = 1; goto read_error;
        }
        if (fread(model->biases[i], sizeof(float), current_layer_size, fp) != (size_t)current_layer_size) {
            error_occurred = 1; goto read_error;
        }
        prev_layer_size = current_layer_size;
    }

    goto cleanup_json; // Success path for parsing

malloc_fail_msg:
    fprintf(stderr, "NoodleNet Error: Memory allocation failed while parsing model.\n");
    goto cleanup_json;
parse_error_msg:
    fprintf(stderr, "NoodleNet Error: Invalid or incomplete JSON metadata structure.\n");
    // error_occurred is already set
    goto cleanup_json;
read_error:
    fprintf(stderr, "NoodleNet Error: Failed to read model file '%s' correctly.\n", model_path);
    // error_occurred is already set
    goto cleanup_json;
parse_error: // cJSON already printed its error
    // error_occurred is already set
    goto cleanup_json;

cleanup_json:
    if (root_json) cJSON_Delete(root_json);
    if (json_str) free(json_str);
    fclose(fp);

    if (error_occurred) {
        free_mlp_model(model); // Clean up any partially allocated model structures
        return -1;
    }
    return 0; // Success
}

static int load_and_process_image(const char* image_path, float* output_buffer) {
    int width, height, channels;
    unsigned char *img_data_orig = stbi_load(image_path, &width, &height, &channels, 0);

    if (!img_data_orig) {
        fprintf(stderr, "NoodleNet Error: Failed to load image '%s'. Reason: %s\n", image_path, stbi_failure_reason());
        return -1;
    }

    if (width == 0 || height == 0) {
         fprintf(stderr, "NoodleNet Error: Image '%s' has zero width or height.\n", image_path);
         stbi_image_free(img_data_orig);
         return -1;
    }

    unsigned char* img_data_resized = NULL;
    unsigned char* current_img_data = img_data_orig;
    int current_width = width;
    int current_height = height;
    int current_channels = channels;

    // Resize if necessary
    if (width != TARGET_WIDTH || height != TARGET_HEIGHT) {
        img_data_resized = (unsigned char*)malloc(TARGET_WIDTH * TARGET_HEIGHT * channels);
        if (!img_data_resized) {
            fprintf(stderr, "NoodleNet Error: Failed to allocate memory for resized image.\n");
            stbi_image_free(img_data_orig);
            return -1;
        }
        // Use bilinear resize function for better quality
        int success = bilinear_resize_image(img_data_orig, width, height, channels,
                                         img_data_resized, TARGET_WIDTH, TARGET_HEIGHT);
        if (!success) {
            fprintf(stderr, "NoodleNet Error: Failed to resize image '%s'.\n", image_path);
            stbi_image_free(img_data_orig);
            free(img_data_resized);
            return -1;
        }
        stbi_image_free(img_data_orig); // Original no longer needed
        current_img_data = img_data_resized;
        current_width = TARGET_WIDTH;
        current_height = TARGET_HEIGHT;
    }

    // Convert to grayscale and normalize
    for (int y = 0; y < TARGET_HEIGHT; ++y) {
        for (int x = 0; x < TARGET_WIDTH; ++x) {
            unsigned char* pixel = current_img_data + (y * TARGET_WIDTH + x) * current_channels;
            float gray_val = 0.0f;

            if (current_channels == 1) { // Already grayscale
                gray_val = (float)pixel[0];
            } else if (current_channels == 2) { // Grayscale + Alpha
                 gray_val = (float)pixel[0];
            } else if (current_channels >= 3) { // RGB or RGBA
                // Standard luminance calculation
                gray_val = 0.299f * pixel[0] + 0.587f * pixel[1] + 0.114f * pixel[2];
            } else {
                 fprintf(stderr, "NoodleNet Error: Unsupported number of channels (%d) in image '%s'.\n", current_channels, image_path);
                 if (img_data_resized) free(img_data_resized); else if (current_img_data != img_data_orig) free(current_img_data);
                 return -1;
            }
            // Normalize to [0, 1]
            output_buffer[y * TARGET_WIDTH + x] = gray_val / 255.0f;
        }
    }

    if (img_data_resized) { // If we allocated for resize
        free(img_data_resized);
    } else if (current_img_data != img_data_orig && img_data_orig != NULL) { // If current_img_data pointed to something else that wasn't img_data_resized
         // This case should not happen with current logic, but defensive.
         // stbi_image_free(img_data_orig) was already called if resized.
         // If not resized, current_img_data is img_data_orig, which is freed here.
         stbi_image_free(img_data_orig);
    } else if (img_data_orig != NULL && current_img_data == img_data_orig) { // If not resized, free original
        stbi_image_free(img_data_orig);
    }

    return 0; // Success
}

static float perform_forward_pass(const MLPModel* model, const float* input_data) {
    float* current_activations = (float*)input_data; // Initially points to input_data
    float* next_activations_buffer = NULL; // To store activations of the next layer
    int prev_layer_num_neurons = model->arch.input_neurons;

    // Iterate through all layers (input -> hidden1 ... -> output)
    for (int layer_idx = 0; layer_idx < model->num_weight_sets; ++layer_idx) {
        int current_layer_num_neurons;
        if (layer_idx < model->arch.num_hidden_layers) {
            current_layer_num_neurons = model->arch.hidden_neurons_per_layer[layer_idx];
        } else { // Output layer
            current_layer_num_neurons = model->arch.output_neurons;
        }

        // Allocate buffer for the current layer's activations (if not the input layer)
        // The output of this loop iteration will be the input for the next
        if (layer_idx < model->num_weight_sets) { // For all layers including output
             next_activations_buffer = (float*)malloc(current_layer_num_neurons * sizeof(float));
             if (!next_activations_buffer) {
                 fprintf(stderr, "NoodleNet Error: Malloc failed for activations buffer.\n");
                 if (current_activations != input_data) free(current_activations); // Free previously allocated buffer
                 return NAN; // Indicate error
             }
        }

        const float* layer_weights = model->weights[layer_idx];
        const float* layer_biases = model->biases[layer_idx];

        for (int j = 0; j < current_layer_num_neurons; ++j) { // For each neuron in current layer
            float weighted_sum = 0.0f;
            // Weights for neuron j are: W_0j, W_1j, ..., W_(prev_layer_num_neurons-1)j
            // These are stored contiguously for neuron j if weights are [prev_neurons][curr_neurons]
            // Or, if weights are [curr_neurons][prev_neurons], then access is different.
            // Assuming weights are stored as a flat array: prev_layer_size * current_layer_size
            // where weight from prev_i to curr_j is at index (j * prev_layer_size + i) OR (i * current_layer_size + j)
            // Let's assume: weights[prev_idx * current_layer_size + current_idx]
            // This means weights for neuron `j` are at `weights[0*CLS+j], weights[1*CLS+j], ...`
            // Or, more commonly: weights from `prev_i` to `curr_j` is at `layer_weights[prev_i * current_layer_num_neurons + j]`

            for (int i = 0; i < prev_layer_num_neurons; ++i) { // For each neuron in previous layer
                // Access weight from neuron i (previous) to neuron j (current)
                weighted_sum += current_activations[i] * layer_weights[i * current_layer_num_neurons + j];
            }
            weighted_sum += layer_biases[j];
            next_activations_buffer[j] = sigmoid(weighted_sum);
        }

        // Free previous layer's activations buffer (if it wasn't the input_data itself)
        if (current_activations != input_data) {
            free(current_activations);
        }
        current_activations = next_activations_buffer; // Current output becomes next input
        next_activations_buffer = NULL; // Buffer ownership transferred
        prev_layer_num_neurons = current_layer_num_neurons;
    }

    // The final `current_activations` buffer holds the output layer's activations.
    // Since output is 1 neuron, this is current_activations[0].
    float final_prediction = current_activations[0];

    // Free the last activations buffer
    if (current_activations != input_data) { // Should always be true unless 0 layers (which is invalid)
        free(current_activations);
    }

    return final_prediction;
}
