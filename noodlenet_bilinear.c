#include "noodlenet.h"
#include "cJSON.h" // For parsing JSON metadata
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // For expf (sigmoid)

// STB Image Loading Library (Single Header)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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
static int load_model_from_file(const char* model_path, MLPModel* model);
static int load_and_process_image(const char* image_path, float* output_buffer);
static float perform_forward_pass(const MLPModel* model, const float* input_data);

// Improved bilinear resize function
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

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void free_mlp_model(MLPModel* model) {
    if (!model) return;

    if (model->arch.hidden_neurons_per_layer) {
        free(model->arch.hidden_neurons_per_layer);
        model->arch.hidden_neurons_per_layer = NULL;
    }

    if (model->weights) {
        for (int i = 0; i < model->num_weight_sets; i++) {
            if (model->weights[i]) free(model->weights[i]);
        }
        free(model->weights);
        model->weights = NULL;
    }

    if (model->biases) {
        for (int i = 0; i < model->num_weight_sets; i++) {
            if (model->biases[i]) free(model->biases[i]);
        }
        free(model->biases);
        model->biases = NULL;
    }

    model->num_weight_sets = 0;
}

// Modified load_and_process_image function to use bilinear interpolation
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
        // Use our bilinear resize function instead of nearest neighbor
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

// The load_model_from_file and perform_forward_pass functions remain the same as in the original noodlenet.c
