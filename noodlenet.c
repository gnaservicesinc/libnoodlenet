// noodlenet.c
// NoodleNet C API Implementation

#include "noodlenet.h"
#include "cJSON.h" // For parsing JSON metadata
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h> // For expf (sigmoid)
#include <time.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

// Apple Accelerate (BLAS) for fast GEMV if available
#if defined(__APPLE__)
#  include <TargetConditionals.h>
#endif
#if defined(__APPLE__)
#  include <Accelerate/Accelerate.h>
#  define NN_USE_ACCELERATE 1
#else
#  define NN_USE_ACCELERATE 0
#endif

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

// Helper function for bicubic interpolation
static float cubic_hermite(float A, float B, float C, float D, float t) {
    float a = -A/2.0f + (3.0f*B)/2.0f - (3.0f*C)/2.0f + D/2.0f;
    float b = A - (5.0f*B)/2.0f + 2.0f*C - D/2.0f;
    float c = -A/2.0f + C/2.0f;
    float d = B;

    return a*t*t*t + b*t*t + c*t + d;
}

// Bicubic resize function (more similar to Qt's smooth transformation)
static int bicubic_resize_image(const unsigned char* src, int src_width, int src_height, int src_channels,
                              unsigned char* dst, int dst_width, int dst_height) {
    float x_ratio = (float)src_width / dst_width;
    float y_ratio = (float)src_height / dst_height;

    for (int y = 0; y < dst_height; y++) {
        float src_y = y * y_ratio;
        int src_y_int = (int)src_y;
        float t_y = src_y - src_y_int;

        for (int x = 0; x < dst_width; x++) {
            float src_x = x * x_ratio;
            int src_x_int = (int)src_x;
            float t_x = src_x - src_x_int;

            for (int c = 0; c < src_channels; c++) {
                float cubic_x[4];

                // For each of the 4 rows around the target pixel
                for (int ky = -1; ky <= 2; ky++) {
                    int py = src_y_int + ky;
                    // Clamp py to valid range
                    py = (py < 0) ? 0 : (py >= src_height) ? src_height - 1 : py;

                    // Get the 4 pixels in this row
                    unsigned char pixels[4];
                    for (int kx = -1; kx <= 2; kx++) {
                        int px = src_x_int + kx;
                        // Clamp px to valid range
                        px = (px < 0) ? 0 : (px >= src_width) ? src_width - 1 : px;
                        pixels[kx+1] = src[(py * src_width + px) * src_channels + c];
                    }

                    // Interpolate horizontally
                    cubic_x[ky+1] = cubic_hermite(pixels[0], pixels[1], pixels[2], pixels[3], t_x);
                }

                // Interpolate vertically
                float result = cubic_hermite(cubic_x[0], cubic_x[1], cubic_x[2], cubic_x[3], t_y);

                // Clamp result to valid range
                result = (result < 0) ? 0 : (result > 255) ? 255 : result;

                dst[(y * dst_width + x) * src_channels + c] = (unsigned char)result;
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

// Layer structure to store activation function pointers
typedef struct {
    ActivationFunction activation_function;
    float (*activate)(float);
    float (*derivative)(float);
} _NoodleNetLayer;

typedef struct {
    ModelArchitecture arch;
    float** weights; // Array of weight matrices (weights[0] = input_to_hidden1, weights[1] = hidden1_to_hidden2, ...)
    float** biases;  // Array of bias vectors
    _NoodleNetLayer* layers; // Array of layer information including activation functions
    int num_weight_sets; // Equal to num_hidden_layers + 1
    // Optimizer state
    NN_Optimizer optimizer;
    float beta1;
    float beta2;
    float epsilon;
    int adam_timestep;
    float** m_weights; // moments for Adam
    float** v_weights; // moments for Adam/RMSprop
    float** m_biases;
    float** v_biases;
    // Scratch buffers to avoid per-sample malloc/free churn
    float** scratch_a;     // a[1..L] (a[0] alias to input)
    float** scratch_z;     // z[0..L-1]
    float** scratch_delta; // delta[0..L-1]
    // Persisted metadata
    char* meta_pos_dir;
    char* meta_neg_dir;
    char* meta_val_dir;
    int   meta_has_locked;
    int   meta_locked_batch_size;
    float meta_locked_learning_rate;
    int   meta_locked_shuffle;
    NN_Optimizer meta_locked_optimizer;
} MLPModel;

// Public opaque alias
struct NN_Model { MLPModel impl; };

// --- Forward Declarations of Static Helper Functions ---
static void free_mlp_model(MLPModel* model);
static float sigmoid(float x);
static int simple_resize_image(const unsigned char* src, int src_width, int src_height, int src_channels,
                              unsigned char* dst, int dst_width, int dst_height);
static int bilinear_resize_image(const unsigned char* src, int src_width, int src_height, int src_channels,
                              unsigned char* dst, int dst_width, int dst_height);
static float cubic_hermite(float A, float B, float C, float D, float t);
static int bicubic_resize_image(const unsigned char* src, int src_width, int src_height, int src_channels,
                              unsigned char* dst, int dst_width, int dst_height);
static int load_model_from_file(const char* model_path, MLPModel* model);
static int load_and_process_image(const char* image_path, float* output_buffer);
static float perform_forward_pass(const MLPModel* model, const float* input_data);
static void set_activation_functions(_NoodleNetLayer* layer, ActivationFunction func_type);
static ActivationFunction parse_activation_function(const char* activation_str);
static const char* activation_to_string(ActivationFunction a);
static void init_random_weights(MLPModel* model);
static void zero_model(MLPModel* model);
static void free_string_array(char** list, int count);
static int list_images_in_dir(const char* dir, char*** out_list, int* out_count);
static float train_one_example(MLPModel* model, const float* x, float y, float lr, float l1, float l2);
static float compute_loss_only(MLPModel* model, const float* x, float y);
static int save_model_to_file(const MLPModel* model, const char* model_path);
static float cosine_similarity(const float* a, const float* b, int n);
static int write_pgm(const char* path, int w, int h, const unsigned char* data);

// Allocate scratch buffers matching the model architecture
static int allocate_scratch(MLPModel* model) {
    int L = model->num_weight_sets;
    // arrays of pointers
    model->scratch_a = (float**)calloc((size_t)L + 1, sizeof(float*)); // we use indices 1..L
    model->scratch_z = (float**)calloc((size_t)L, sizeof(float*));
    model->scratch_delta = (float**)calloc((size_t)L, sizeof(float*));
    if (!model->scratch_a || !model->scratch_z || !model->scratch_delta) return -1;
    int prev = model->arch.input_neurons;
    for (int i = 0; i < L; ++i) {
        int curr = (i < model->arch.num_hidden_layers) ? model->arch.hidden_neurons_per_layer[i]
                                                       : model->arch.output_neurons;
        model->scratch_z[i] = (float*)malloc(sizeof(float) * (size_t)curr);
        model->scratch_a[i+1] = (float*)malloc(sizeof(float) * (size_t)curr);
        model->scratch_delta[i] = (float*)malloc(sizeof(float) * (size_t)curr);
        if (!model->scratch_z[i] || !model->scratch_a[i+1] || !model->scratch_delta[i]) return -1;
        prev = curr;
    }
    return 0;
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

    if (model->layers) {
        free(model->layers);
        model->layers = NULL;
    }

    // Free scratch buffers
    if (model->scratch_a) {
        for (int i = 1; i <= model->num_weight_sets; ++i) {
            if (model->scratch_a[i]) free(model->scratch_a[i]);
        }
        free(model->scratch_a);
        model->scratch_a = NULL;
    }
    if (model->scratch_z) {
        for (int i = 0; i < model->num_weight_sets; ++i) {
            if (model->scratch_z[i]) free(model->scratch_z[i]);
        }
        free(model->scratch_z);
        model->scratch_z = NULL;
    }
    if (model->scratch_delta) {
        for (int i = 0; i < model->num_weight_sets; ++i) {
            if (model->scratch_delta[i]) free(model->scratch_delta[i]);
        }
        free(model->scratch_delta);
        model->scratch_delta = NULL;
    }

    if (model->m_weights) {
        for (int i = 0; i < model->num_weight_sets; ++i) if (model->m_weights[i]) free(model->m_weights[i]);
        free(model->m_weights); model->m_weights = NULL;
    }
    if (model->v_weights) {
        for (int i = 0; i < model->num_weight_sets; ++i) if (model->v_weights[i]) free(model->v_weights[i]);
        free(model->v_weights); model->v_weights = NULL;
    }
    if (model->m_biases) {
        for (int i = 0; i < model->arch.num_hidden_layers + 1; ++i) if (model->m_biases[i]) free(model->m_biases[i]);
        free(model->m_biases); model->m_biases = NULL;
    }
    if (model->v_biases) {
        for (int i = 0; i < model->arch.num_hidden_layers + 1; ++i) if (model->v_biases[i]) free(model->v_biases[i]);
        free(model->v_biases); model->v_biases = NULL;
    }

    model->num_weight_sets = 0;

    // Free metadata strings
    if (model->meta_pos_dir) { free(model->meta_pos_dir); model->meta_pos_dir = NULL; }
    if (model->meta_neg_dir) { free(model->meta_neg_dir); model->meta_neg_dir = NULL; }
    if (model->meta_val_dir) { free(model->meta_val_dir); model->meta_val_dir = NULL; }
    model->meta_has_locked = 0;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

// Tanh activation function
static float tanh_activate(float x) {
    return tanhf(x); // Use the standard library's tanhf for efficiency and precision
}

static float tanh_derivative(float x) {
    float th = tanhf(x);
    return 1.0f - (th * th);
}

// ReLU activation function
static float relu_activate(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

static float relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

// Leaky ReLU activation function
#define LEAKY_RELU_SLOPE 0.01f

static float leaky_relu_activate(float x) {
    return (x > 0.0f) ? x : LEAKY_RELU_SLOPE * x;
}

static float leaky_relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : LEAKY_RELU_SLOPE;
}

// Helper function to set activation functions based on enum
static void set_activation_functions(
    _NoodleNetLayer* layer,
    ActivationFunction func_type
) {
    layer->activation_function = func_type; // Store the enum type

    switch (func_type) {
        case NN_ACTIVATION_FUNCTION_TANH:
            layer->activate = tanh_activate;
            layer->derivative = tanh_derivative;
            break;
        case NN_ACTIVATION_FUNCTION_RELU:
            layer->activate = relu_activate;
            layer->derivative = relu_derivative;
            break;
        case NN_ACTIVATION_FUNCTION_LEAKY_RELU:
            layer->activate = leaky_relu_activate;
            layer->derivative = leaky_relu_derivative;
            break;
        case NN_ACTIVATION_FUNCTION_SIGMOID:
        default:
            layer->activate = sigmoid;
            layer->derivative = sigmoid_derivative;
            break;
    }
}

// Helper function to parse activation function string to enum
static ActivationFunction parse_activation_function(const char* activation_str) {
    if (strcmp(activation_str, "tanh") == 0) {
        return NN_ACTIVATION_FUNCTION_TANH;
    } else if (strcmp(activation_str, "relu") == 0) {
        return NN_ACTIVATION_FUNCTION_RELU;
    } else if (strcmp(activation_str, "leaky_relu") == 0) {
        return NN_ACTIVATION_FUNCTION_LEAKY_RELU;
    } else {
        return NN_ACTIVATION_FUNCTION_SIGMOID; // Default
    }
}

static const char* activation_to_string(ActivationFunction a) {
    switch (a) {
        case NN_ACTIVATION_FUNCTION_TANH: return "tanh";
        case NN_ACTIVATION_FUNCTION_RELU: return "relu";
        case NN_ACTIVATION_FUNCTION_LEAKY_RELU: return "leaky_relu";
        case NN_ACTIVATION_FUNCTION_SIGMOID:
        default: return "sigmoid";
    }
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

    // Parse output activation function
    cJSON* output_act_json = cJSON_GetObjectItemCaseSensitive(arch_json, "output_activation");
    if (!cJSON_IsString(output_act_json)) {
        fprintf(stderr, "NoodleNet Error: Model output activation function must be specified.\n");
        error_occurred = 1; goto parse_error_msg;
    }
    ActivationFunction output_activation = parse_activation_function(output_act_json->valuestring);

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

    // Allocate memory for layer information (hidden layers + output layer)
    model->layers = (_NoodleNetLayer*)malloc((model->arch.num_hidden_layers + 1) * sizeof(_NoodleNetLayer));
    if (!model->layers && (model->arch.num_hidden_layers + 1) > 0) {
        error_occurred = 1; goto malloc_fail_msg;
    }

    for (int i = 0; i < model->arch.num_hidden_layers; ++i) {
        cJSON* layer_item = cJSON_GetArrayItem(hidden_layers_json, i);
        int neurons = 0;
        ActivationFunction layer_activation = NN_ACTIVATION_FUNCTION_SIGMOID; // Default

        if (version_read == 2) {
            // New format: each item is an object with "neurons" and "activation"
            if (!cJSON_IsObject(layer_item)) { error_occurred = 1; goto parse_error_msg; }

            cJSON* neurons_json = cJSON_GetObjectItemCaseSensitive(layer_item, "neurons");
            if (!cJSON_IsNumber(neurons_json) || neurons_json->valueint <= 0) {
                error_occurred = 1; goto parse_error_msg;
            }
            neurons = neurons_json->valueint;

            // Parse activation function
            cJSON* act_json = cJSON_GetObjectItemCaseSensitive(layer_item, "activation");
            if (cJSON_IsString(act_json)) {
                layer_activation = parse_activation_function(act_json->valuestring);
            }
        } else {
            // Old format: each item is just the number of neurons (default to sigmoid)
            if (!cJSON_IsNumber(layer_item) || layer_item->valueint <= 0) {
                error_occurred = 1; goto parse_error_msg;
            }
            neurons = layer_item->valueint;
            layer_activation = NN_ACTIVATION_FUNCTION_SIGMOID;
        }

        model->arch.hidden_neurons_per_layer[i] = neurons;
        // Set activation function for this hidden layer
        set_activation_functions(&model->layers[i], layer_activation);
    }

    // Set activation function for output layer
    set_activation_functions(&model->layers[model->arch.num_hidden_layers], output_activation);

    // --- Optional metadata ---
    cJSON* data_dirs = cJSON_GetObjectItemCaseSensitive(root_json, "data_dirs");
    if (cJSON_IsObject(data_dirs)) {
        cJSON* p = cJSON_GetObjectItemCaseSensitive(data_dirs, "positive");
        cJSON* n = cJSON_GetObjectItemCaseSensitive(data_dirs, "negative");
        cJSON* v = cJSON_GetObjectItemCaseSensitive(data_dirs, "validation");
        // Only keep dirs that exist
        struct stat st;
        if (cJSON_IsString(p) && p->valuestring && stat(p->valuestring, &st) == 0 && S_ISDIR(st.st_mode)) model->meta_pos_dir = strdup(p->valuestring);
        if (cJSON_IsString(n) && n->valuestring && stat(n->valuestring, &st) == 0 && S_ISDIR(st.st_mode)) model->meta_neg_dir = strdup(n->valuestring);
        if (cJSON_IsString(v) && v->valuestring && stat(v->valuestring, &st) == 0 && S_ISDIR(st.st_mode)) model->meta_val_dir = strdup(v->valuestring);
    }
    cJSON* lock = cJSON_GetObjectItemCaseSensitive(root_json, "training_locked_params");
    if (cJSON_IsObject(lock)) {
        cJSON* bs = cJSON_GetObjectItemCaseSensitive(lock, "batch_size");
        cJSON* lr = cJSON_GetObjectItemCaseSensitive(lock, "learning_rate");
        cJSON* sh = cJSON_GetObjectItemCaseSensitive(lock, "shuffle");
        cJSON* opt = cJSON_GetObjectItemCaseSensitive(lock, "optimizer");
        if (cJSON_IsNumber(bs) && cJSON_IsNumber(lr) && (cJSON_IsNumber(sh) || cJSON_IsBool(sh))) {
            model->meta_has_locked = 1;
            model->meta_locked_batch_size = bs->valueint;
            model->meta_locked_learning_rate = (float)lr->valuedouble;
            model->meta_locked_shuffle = cJSON_IsNumber(sh) ? sh->valueint : (cJSON_IsTrue(sh) ? 1 : 0);
            if (cJSON_IsString(opt) && opt->valuestring) {
                if (strcmp(opt->valuestring, "Adam") == 0) model->meta_locked_optimizer = NN_OPTIMIZER_ADAM;
                else if (strcmp(opt->valuestring, "RMSprop") == 0) model->meta_locked_optimizer = NN_OPTIMIZER_RMSPROP;
                else model->meta_locked_optimizer = NN_OPTIMIZER_SGD;
            } else {
                model->meta_locked_optimizer = NN_OPTIMIZER_SGD;
            }
        }
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
        // Use bicubic resize function for better quality (more similar to Qt's smooth transformation)
        int success = bicubic_resize_image(img_data_orig, width, height, channels,
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

// Initialize model weights with Xavier uniform and zero biases
static void init_random_weights(MLPModel* model) {
    srand((unsigned)time(NULL));
    int prev = model->arch.input_neurons;
    for (int i = 0; i < model->num_weight_sets; ++i) {
        int curr = (i < model->arch.num_hidden_layers) ? model->arch.hidden_neurons_per_layer[i]
                                                       : model->arch.output_neurons;
        long n = (long)prev * curr;
        float limit = sqrtf(6.0f / (prev + curr));
        for (long k = 0; k < n; ++k) {
            float r = (float)rand() / (float)RAND_MAX; // [0,1]
            model->weights[i][k] = -limit + 2.0f * limit * r;
        }
        for (int j = 0; j < curr; ++j) {
            model->biases[i][j] = 0.0f;
        }
        prev = curr;
    }
}

static void zero_model(MLPModel* model) {
    if (!model) return;
    memset(model, 0, sizeof(*model));
}

static void free_string_array(char** list, int count) {
    if (!list) return;
    for (int i = 0; i < count; ++i) free(list[i]);
    free(list);
}

static int ends_with_case(const char* s, const char* suf) {
    size_t n = strlen(s), m = strlen(suf);
    if (m > n) return 0;
    const char* a = s + (n - m);
    for (size_t i = 0; i < m; ++i) {
        char c1 = a[i];
        char c2 = suf[i];
        if (c1 >= 'A' && c1 <= 'Z') c1 = (char)(c1 - 'A' + 'a');
        if (c2 >= 'A' && c2 <= 'Z') c2 = (char)(c2 - 'A' + 'a');
        if (c1 != c2) return 0;
    }
    return 1;
}

static int list_images_in_dir(const char* dir, char*** out_list, int* out_count) {
    *out_list = NULL; *out_count = 0;
    DIR* d = opendir(dir);
    if (!d) return -1;
    struct dirent* ent;
    int cap = 32;
    char** list = (char**)malloc(sizeof(char*) * cap);
    if (!list) { closedir(d); return -1; }
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) continue;
        if (!(ends_with_case(path, ".png") || ends_with_case(path, ".bmp") || ends_with_case(path, ".jpg") || ends_with_case(path, ".jpeg")))
            continue;
        if (*out_count >= cap) {
            cap *= 2;
            char** tmp = (char**)realloc(list, sizeof(char*) * cap);
            if (!tmp) { free_string_array(list, *out_count); closedir(d); return -1; }
            list = tmp;
        }
        list[*out_count] = strdup(path);
        if (!list[*out_count]) { free_string_array(list, *out_count); closedir(d); return -1; }
        (*out_count)++;
    }
    closedir(d);
    *out_list = list;
    return 0;
}

// Train a single (x,y) example using SGD; returns loss
static float train_one_example(MLPModel* model, const float* x, float y, float lr, float l1, float l2) {
    int L = model->num_weight_sets; // number of layers with parameters
    if (model->optimizer == NN_OPTIMIZER_ADAM) {
        model->adam_timestep += 1;
    }

    // Use preallocated scratch buffers
    float** a = model->scratch_a;     // a[0] set to x below
    float** z = model->scratch_z;
    float** delta = model->scratch_delta;
    if (!a || !z || !delta) return NAN;

    int prev = model->arch.input_neurons;
    a[0] = (float*)x; // alias input

    // Forward: z = W*a + b; a = act(z)
    for (int i = 0; i < L; ++i) {
        int curr = (i < model->arch.num_hidden_layers) ? model->arch.hidden_neurons_per_layer[i]
                                                       : model->arch.output_neurons;
        float* Wi = model->weights[i];
        float* bi = model->biases[i];
        float* zi = z[i];
        float* ai_next = a[i+1];
#if NN_USE_ACCELERATE
        // Copy bias into z, then GEMV accumulate W*a into z
        memcpy(zi, bi, sizeof(float) * (size_t)curr);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, curr, prev, 1.0f, Wi, prev, a[i], 1, 1.0f, zi, 1);
        for (int r = 0; r < curr; ++r) ai_next[r] = model->layers[i].activate(zi[r]);
#else
        for (int r = 0; r < curr; ++r) {
            double sum = bi[r];
            const float* wrow = Wi + (long)r * prev;
            for (int c = 0; c < prev; ++c) sum += wrow[c] * a[i][c];
            zi[r] = (float)sum;
            ai_next[r] = model->layers[i].activate(zi[r]);
        }
#endif
        prev = curr;
    }

    // Loss (binary cross-entropy) using output activation a[L][0]
    float o = a[L][0];
    if (o < 1e-7f) o = 1e-7f; if (o > 1.0f - 1e-7f) o = 1.0f - 1e-7f;
    float loss = -(y * logf(o) + (1.0f - y) * logf(1.0f - o));

    // Backward
    // Output delta: dL/da * da/dz
    // dL/da = -(y/o) + (1-y)/(1-o)
    float dL_da = -(y / o) + ((1.0f - y) / (1.0f - o));
    float zL = z[L-1][0];
    float da_dz;
    if (model->layers[L-1].derivative) da_dz = model->layers[L-1].derivative(zL);
    else { float s = 1.0f / (1.0f + expf(-zL)); da_dz = s * (1.0f - s); }
    delta[L-1][0] = dL_da * da_dz;

    // Hidden deltas: delta[i] = (W[i+1]^T * delta[i+1]) .* act'(z[i])
    for (int i = L - 2; i >= 0; --i) {
        int curr = (i < model->arch.num_hidden_layers) ? model->arch.hidden_neurons_per_layer[i]
                                                       : model->arch.output_neurons; // not used
        int next = (i+1 < model->arch.num_hidden_layers) ? model->arch.hidden_neurons_per_layer[i+1]
                                                         : model->arch.output_neurons;
        float* Wnext = model->weights[i+1];
#if NN_USE_ACCELERATE
        // delta[i] = Wnext^T * delta[i+1]
        cblas_sgemv(CblasRowMajor, CblasTrans, next, curr, 1.0f, Wnext, curr, delta[i+1], 1, 0.0f, delta[i], 1);
        // elementwise multiply by derivative
        for (int r = 0; r < curr; ++r) {
            float dz = model->layers[i].derivative(z[i][r]);
            delta[i][r] = delta[i][r] * dz;
        }
#else
        for (int r = 0; r < curr; ++r) {
            double sum = 0.0;
            for (int j = 0; j < next; ++j) {
                sum += Wnext[(long)j * curr + r] * delta[i+1][j];
            }
            float dz = model->layers[i].derivative(z[i][r]);
            delta[i][r] = (float)(sum * dz);
        }
#endif
    }

    // Update weights depending on optimizer (SGD/RMSprop/Adam) with L1/L2 on weights
    prev = model->arch.input_neurons;
    for (int i = 0; i < L; ++i) {
        int curr = (i < model->arch.num_hidden_layers) ? model->arch.hidden_neurons_per_layer[i]
                                                       : model->arch.output_neurons;
        float* Wi = model->weights[i];
        float* bi = model->biases[i];
        float* mW = model->m_weights ? model->m_weights[i] : NULL;
        float* vW = model->v_weights ? model->v_weights[i] : NULL;
        for (int r = 0; r < curr; ++r) {
            // bias grad
            float g_b = delta[i][r];
            if (model->optimizer == NN_OPTIMIZER_SGD) {
                bi[r] -= lr * g_b;
            } else if (model->optimizer == NN_OPTIMIZER_RMSPROP) {
                float* vB = model->v_biases[i];
                vB[r] = model->beta2 * vB[r] + (1.0f - model->beta2) * g_b * g_b;
                bi[r] -= lr * g_b / (sqrtf(vB[r]) + model->epsilon);
            } else { // Adam
                float* mB = model->m_biases[i];
                float* vB = model->v_biases[i];
                mB[r] = model->beta1 * mB[r] + (1.0f - model->beta1) * g_b;
                vB[r] = model->beta2 * vB[r] + (1.0f - model->beta2) * (g_b * g_b);
                float b1t = powf(model->beta1, (float)(model->adam_timestep + 1));
                float b2t = powf(model->beta2, (float)(model->adam_timestep + 1));
                float mhat = mB[r] / (1.0f - b1t);
                float vhat = vB[r] / (1.0f - b2t);
                bi[r] -= lr * mhat / (sqrtf(vhat) + model->epsilon);
            }

            // weight row update
            float* wrow = Wi + (long)r * prev;
            if (model->optimizer == NN_OPTIMIZER_SGD) {
                // Apply L2 via scaling, then gradient via AXPY, then L1 via loop
#if NN_USE_ACCELERATE
                if (l2 > 0.0f) {
                    float scale = 1.0f - lr * l2;
                    cblas_sscal(prev, scale, wrow, 1);
                }
                // wrow += (-lr * delta) * a[i]
                cblas_saxpy(prev, -lr * delta[i][r], a[i], 1, wrow, 1);
                if (l1 > 0.0f) {
                    float step = lr * l1;
                    for (int c = 0; c < prev; ++c) {
                        float s = (wrow[c] > 0.0f ? 1.0f : (wrow[c] < 0.0f ? -1.0f : 0.0f));
                        wrow[c] -= step * s;
                    }
                }
#else
                for (int c = 0; c < prev; ++c) {
                    float grad = delta[i][r] * a[i][c];
                    if (l2 > 0.0f) grad += l2 * wrow[c];
                    if (l1 > 0.0f) grad += l1 * (wrow[c] > 0.0f ? 1.0f : (wrow[c] < 0.0f ? -1.0f : 0.0f));
                    wrow[c] -= lr * grad;
                }
#endif
            } else if (model->optimizer == NN_OPTIMIZER_RMSPROP) {
                for (int c = 0; c < prev; ++c) {
                    float grad = delta[i][r] * a[i][c];
                    if (l2 > 0.0f) grad += l2 * wrow[c];
                    if (l1 > 0.0f) grad += l1 * (wrow[c] > 0.0f ? 1.0f : (wrow[c] < 0.0f ? -1.0f : 0.0f));
                    vW[(long)r * prev + c] = model->beta2 * vW[(long)r * prev + c] + (1.0f - model->beta2) * grad * grad;
                    wrow[c] -= lr * grad / (sqrtf(vW[(long)r * prev + c]) + model->epsilon);
                }
            } else { // Adam
                for (int c = 0; c < prev; ++c) {
                    float grad = delta[i][r] * a[i][c];
                    if (l2 > 0.0f) grad += l2 * wrow[c];
                    if (l1 > 0.0f) grad += l1 * (wrow[c] > 0.0f ? 1.0f : (wrow[c] < 0.0f ? -1.0f : 0.0f));
                    mW[(long)r * prev + c] = model->beta1 * mW[(long)r * prev + c] + (1.0f - model->beta1) * grad;
                    vW[(long)r * prev + c] = model->beta2 * vW[(long)r * prev + c] + (1.0f - model->beta2) * grad * grad;
                    float b1t = powf(model->beta1, (float)(model->adam_timestep + 1));
                    float b2t = powf(model->beta2, (float)(model->adam_timestep + 1));
                    float mhat = mW[(long)r * prev + c] / (1.0f - b1t);
                    float vhat = vW[(long)r * prev + c] / (1.0f - b2t);
                    wrow[c] -= lr * mhat / (sqrtf(vhat) + model->epsilon);
                }
            }
        }
        prev = curr;
    }

    return loss;
}

// Compute loss for (x,y) without updating parameters
static float compute_loss_only(MLPModel* model, const float* x, float y) {
    int L = model->num_weight_sets;
    float** a = model->scratch_a;
    float** z = model->scratch_z;
    if (!a || !z) return NAN;
    a[0] = (float*)x; int prev = model->arch.input_neurons;
    for (int i=0;i<L;++i) {
        int curr = (i < model->arch.num_hidden_layers) ? model->arch.hidden_neurons_per_layer[i]
                                                       : model->arch.output_neurons;
        const float* Wi = model->weights[i]; const float* bi = model->biases[i];
#if NN_USE_ACCELERATE
        memcpy(z[i], bi, sizeof(float) * (size_t)curr);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, curr, prev, 1.0f, Wi, prev, a[i], 1, 1.0f, z[i], 1);
        for (int r=0;r<curr;++r) a[i+1][r] = model->layers[i].activate(z[i][r]);
#else
        for (int r=0;r<curr;++r){ double sum=bi[r]; const float* wrow = Wi + (long)r * prev; for (int c=0;c<prev;++c) sum += wrow[c]*a[i][c]; z[i][r] = (float)sum; a[i+1][r] = model->layers[i].activate(z[i][r]); }
#endif
        prev = curr;
    }
    float o = a[L][0]; if (o < 1e-7f) o = 1e-7f; if (o > 1.0f - 1e-7f) o = 1.0f - 1e-7f;
    float loss = -(y * logf(o) + (1.0f - y) * logf(1.0f - o));
    return loss;
}

static int save_model_to_file(const MLPModel* model, const char* model_path) {
    FILE* fp = fopen(model_path, "wb");
    if (!fp) return -1;
    // Magic + version
    unsigned int magic = MODEL_MAGIC_NUMBER;
    unsigned char version = MODEL_FORMAT_VERSION;
    fwrite(&magic, sizeof(magic), 1, fp);
    fwrite(&version, sizeof(version), 1, fp);

    // Build JSON metadata
    cJSON* root = cJSON_CreateObject();
    cJSON* arch = cJSON_CreateObject();
    cJSON_AddItemToObject(root, "architecture", arch);
    cJSON_AddNumberToObject(arch, "input_neurons", model->arch.input_neurons);
    cJSON_AddNumberToObject(arch, "output_neurons", model->arch.output_neurons);
    cJSON_AddStringToObject(root, "data_precision", "float");

    // Hidden layers
    cJSON* hidden = cJSON_CreateArray();
    for (int i = 0; i < model->arch.num_hidden_layers; ++i) {
        cJSON* hl = cJSON_CreateObject();
        cJSON_AddNumberToObject(hl, "neurons", model->arch.hidden_neurons_per_layer[i]);
        cJSON_AddStringToObject(hl, "activation", activation_to_string(model->layers[i].activation_function));
        cJSON_AddItemToArray(hidden, hl);
    }
    cJSON_AddItemToObject(arch, "hidden_layers", hidden);
    cJSON_AddStringToObject(arch, "output_activation", activation_to_string(model->layers[model->arch.num_hidden_layers].activation_function));

    // Append metadata: data_dirs and training_locked_params if available
    if (model->meta_pos_dir || model->meta_neg_dir || model->meta_val_dir) {
        cJSON* data_dirs = cJSON_CreateObject();
        if (model->meta_pos_dir) cJSON_AddStringToObject(data_dirs, "positive", model->meta_pos_dir);
        if (model->meta_neg_dir) cJSON_AddStringToObject(data_dirs, "negative", model->meta_neg_dir);
        if (model->meta_val_dir) cJSON_AddStringToObject(data_dirs, "validation", model->meta_val_dir);
        cJSON_AddItemToObject(root, "data_dirs", data_dirs);
    }
    if (model->meta_has_locked) {
        cJSON* lock = cJSON_CreateObject();
        cJSON_AddNumberToObject(lock, "batch_size", model->meta_locked_batch_size);
        cJSON_AddNumberToObject(lock, "learning_rate", model->meta_locked_learning_rate);
        cJSON_AddNumberToObject(lock, "shuffle", model->meta_locked_shuffle);
        const char* oname = (model->meta_locked_optimizer==NN_OPTIMIZER_ADAM?"Adam":(model->meta_locked_optimizer==NN_OPTIMIZER_RMSPROP?"RMSprop":"SGD"));
        cJSON_AddStringToObject(lock, "optimizer", oname);
        cJSON_AddItemToObject(root, "training_locked_params", lock);
    }

    char* json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!json_str) { fclose(fp); return -1; }
    unsigned int len = (unsigned int)strlen(json_str);
    fwrite(&len, sizeof(len), 1, fp);
    fwrite(json_str, 1, len, fp);
    free(json_str);

    // Write weights and biases
    int prev = model->arch.input_neurons;
    for (int i = 0; i < model->num_weight_sets; ++i) {
        int curr = (i < model->arch.num_hidden_layers) ? model->arch.hidden_neurons_per_layer[i]
                                                       : model->arch.output_neurons;
        long n = (long)prev * curr;
        fwrite(model->weights[i], sizeof(float), n, fp);
        fwrite(model->biases[i], sizeof(float), curr, fp);
        prev = curr;
    }

    fclose(fp);
    return 0;
}

static float cosine_similarity(const float* a, const float* b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; ++i) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;
    double cs = dot / (sqrt(na) * sqrt(nb));
    if (cs < -1.0) cs = -1.0; if (cs > 1.0) cs = 1.0;
    return (float)cs;
}

static int write_pgm(const char* path, int w, int h, const unsigned char* data) {
    FILE* fp = fopen(path, "wb");
    if (!fp) return -1;
    fprintf(fp, "P5\n%d %d\n255\n", w, h);
    fwrite(data, 1, (size_t)(w*h), fp);
    fclose(fp);
    return 0;
}

// -------- Public API Extensions --------

NN_Model* nn_model_create(int input_neurons,
                          const int* hidden_sizes, int num_hidden,
                          int output_neurons,
                          const ActivationFunction* hidden_activations,
                          ActivationFunction output_activation) {
    NN_Model* m = (NN_Model*)calloc(1, sizeof(NN_Model));
    if (!m) return NULL;
    MLPModel* model = &m->impl;
    model->arch.input_neurons = input_neurons;
    model->arch.output_neurons = output_neurons;
    model->arch.num_hidden_layers = num_hidden;
    model->arch.hidden_neurons_per_layer = NULL;
    if (num_hidden > 0) {
        model->arch.hidden_neurons_per_layer = (int*)malloc(sizeof(int)*num_hidden);
        if (!model->arch.hidden_neurons_per_layer) { nn_model_free(m); return NULL; }
        for (int i=0;i<num_hidden;++i) model->arch.hidden_neurons_per_layer[i] = hidden_sizes[i];
    }
    model->num_weight_sets = num_hidden + 1;
    model->weights = (float**)calloc(model->num_weight_sets, sizeof(float*));
    model->biases  = (float**)calloc(model->num_weight_sets, sizeof(float*));
    model->layers  = (_NoodleNetLayer*)calloc(model->num_weight_sets, sizeof(_NoodleNetLayer));
    if (!model->weights || !model->biases || !model->layers) { nn_model_free(m); return NULL; }

    int prev = input_neurons;
    for (int i = 0; i < model->num_weight_sets; ++i) {
        int curr = (i < num_hidden) ? hidden_sizes[i] : output_neurons;
        long n = (long)prev * curr;
        model->weights[i] = (float*)malloc(sizeof(float)*n);
        model->biases[i]  = (float*)malloc(sizeof(float)*curr);
        if (!model->weights[i] || !model->biases[i]) { nn_model_free(m); return NULL; }
        prev = curr;
    }
    for (int i=0;i<num_hidden;++i) set_activation_functions(&model->layers[i], hidden_activations ? hidden_activations[i] : NN_ACTIVATION_FUNCTION_SIGMOID);
    set_activation_functions(&model->layers[num_hidden], output_activation);
    init_random_weights(model);
    // Initialize optimizer defaults and allocate state arrays
    model->optimizer = NN_OPTIMIZER_SGD;
    model->beta1 = 0.9f; model->beta2 = 0.999f; model->epsilon = 1e-8f; model->adam_timestep = 0;
    model->m_weights = (float**)calloc(model->num_weight_sets, sizeof(float*));
    model->v_weights = (float**)calloc(model->num_weight_sets, sizeof(float*));
    model->m_biases  = (float**)calloc(model->arch.num_hidden_layers + 1, sizeof(float*));
    model->v_biases  = (float**)calloc(model->arch.num_hidden_layers + 1, sizeof(float*));
    if (!model->m_weights || !model->v_weights || !model->m_biases || !model->v_biases) { nn_model_free(m); return NULL; }
    int prev_in = input_neurons;
    for (int i=0;i<model->num_weight_sets;++i) {
        int curr = (i < num_hidden) ? hidden_sizes[i] : output_neurons;
        long n = (long)prev_in * curr;
        model->m_weights[i] = (float*)calloc(n, sizeof(float));
        model->v_weights[i] = (float*)calloc(n, sizeof(float));
        if (!model->m_weights[i] || !model->v_weights[i]) { nn_model_free(m); return NULL; }
        prev_in = curr;
    }
    for (int i=0;i<model->arch.num_hidden_layers+1;++i) {
        int len = (i<model->arch.num_hidden_layers)? model->arch.hidden_neurons_per_layer[i] : model->arch.output_neurons;
        model->m_biases[i] = (float*)calloc(len, sizeof(float));
        model->v_biases[i] = (float*)calloc(len, sizeof(float));
        if (!model->m_biases[i] || !model->v_biases[i]) { nn_model_free(m); return NULL; }
    }
    // Allocate scratch buffers for faster per-sample training
    if (allocate_scratch(model) != 0) { nn_model_free(m); return NULL; }
    return m;
}

NN_Model* nn_model_load(const char* model_path) {
    NN_Model* m = (NN_Model*)calloc(1, sizeof(NN_Model));
    if (!m) return NULL;
    if (load_model_from_file(model_path, &m->impl) != 0) { free(m); return NULL; }
    // Initialize optimizer state arrays for a loaded model
    MLPModel* model = &m->impl;
    model->optimizer = NN_OPTIMIZER_SGD;
    model->beta1 = 0.9f; model->beta2 = 0.999f; model->epsilon = 1e-8f; model->adam_timestep = 0;
    model->m_weights = (float**)calloc(model->num_weight_sets, sizeof(float*));
    model->v_weights = (float**)calloc(model->num_weight_sets, sizeof(float*));
    model->m_biases  = (float**)calloc(model->arch.num_hidden_layers + 1, sizeof(float*));
    model->v_biases  = (float**)calloc(model->arch.num_hidden_layers + 1, sizeof(float*));
    if (!model->m_weights || !model->v_weights || !model->m_biases || !model->v_biases) { nn_model_free(m); return NULL; }
    int prev_in = model->arch.input_neurons;
    for (int i=0;i<model->num_weight_sets;++i) {
        int curr = (i<model->arch.num_hidden_layers)? model->arch.hidden_neurons_per_layer[i] : model->arch.output_neurons;
        long n = (long)prev_in * curr;
        model->m_weights[i] = (float*)calloc(n, sizeof(float));
        model->v_weights[i] = (float*)calloc(n, sizeof(float));
        if (!model->m_weights[i] || !model->v_weights[i]) { nn_model_free(m); return NULL; }
        prev_in = curr;
    }
    for (int i=0;i<model->arch.num_hidden_layers+1;++i) {
        int len = (i<model->arch.num_hidden_layers)? model->arch.hidden_neurons_per_layer[i] : model->arch.output_neurons;
        model->m_biases[i] = (float*)calloc(len, sizeof(float));
        model->v_biases[i] = (float*)calloc(len, sizeof(float));
        if (!model->m_biases[i] || !model->v_biases[i]) { nn_model_free(m); return NULL; }
    }
    // Allocate scratch buffers for faster per-sample training
    if (allocate_scratch(model) != 0) { nn_model_free(m); return NULL; }
    return m;
}

int nn_model_save(const NN_Model* model, const char* model_path) {
    if (!model) return -1;
    return save_model_to_file(&model->impl, model_path);
}

void nn_model_free(NN_Model* model) {
    if (!model) return;
    free_mlp_model(&model->impl);
    free(model);
}

int nn_model_num_hidden(const NN_Model* model) { return model ? model->impl.arch.num_hidden_layers : -1; }
int nn_model_hidden_size(const NN_Model* model, int layer_index) {
    if (!model) return -1;
    if (layer_index < 0 || layer_index >= model->impl.arch.num_hidden_layers) return -1;
    return model->impl.arch.hidden_neurons_per_layer[layer_index];
}
int nn_model_input_size(const NN_Model* model) { return model ? model->impl.arch.input_neurons : -1; }
int nn_model_output_size(const NN_Model* model) { return model ? model->impl.arch.output_neurons : -1; }
ActivationFunction nn_model_hidden_activation(const NN_Model* model, int layer_index) {
    if (!model) return NN_ACTIVATION_FUNCTION_SIGMOID;
    if (layer_index < 0 || layer_index >= model->impl.arch.num_hidden_layers) return NN_ACTIVATION_FUNCTION_SIGMOID;
    return model->impl.layers[layer_index].activation_function;
}
ActivationFunction nn_model_output_activation(const NN_Model* model) {
    if (!model) return NN_ACTIVATION_FUNCTION_SIGMOID;
    return model->impl.layers[model->impl.arch.num_hidden_layers].activation_function;
}

int nn_model_predict_image(const NN_Model* model, const char* image_path, float* out_prob) {
    if (!model || !out_prob) return -1;
    float* input = (float*)malloc(sizeof(float) * EXPECTED_INPUT_NEURONS);
    if (!input) return -1;
    if (load_and_process_image(image_path, input) != 0) { free(input); return -1; }
    float p = perform_forward_pass(&model->impl, input);
    free(input);
    if (isnan(p) || isinf(p)) return -1;
    if (p < 0.0f) p = 0.0f; if (p > 1.0f) p = 1.0f;
    *out_prob = p;
    return 0;
}

// --- Metadata helpers ---
int nn_model_set_data_dirs(NN_Model* model,
                           const char* pos_dir,
                           const char* neg_dir,
                           const char* val_dir) {
    if (!model) return -1;
    MLPModel* m = &model->impl;
    if (m->meta_pos_dir) { free(m->meta_pos_dir); m->meta_pos_dir = NULL; }
    if (m->meta_neg_dir) { free(m->meta_neg_dir); m->meta_neg_dir = NULL; }
    if (m->meta_val_dir) { free(m->meta_val_dir); m->meta_val_dir = NULL; }
    if (pos_dir && *pos_dir) m->meta_pos_dir = strdup(pos_dir);
    if (neg_dir && *neg_dir) m->meta_neg_dir = strdup(neg_dir);
    if (val_dir && *val_dir) m->meta_val_dir = strdup(val_dir);
    return 0;
}

int nn_model_get_data_dirs(const NN_Model* model,
                           const char** out_pos_dir,
                           const char** out_neg_dir,
                           const char** out_val_dir) {
    if (!model) return -1;
    if (out_pos_dir) *out_pos_dir = model->impl.meta_pos_dir;
    if (out_neg_dir) *out_neg_dir = model->impl.meta_neg_dir;
    if (out_val_dir) *out_val_dir = model->impl.meta_val_dir;
    return 0;
}

int nn_model_set_locked_training_params(NN_Model* model,
                                        int batch_size,
                                        float learning_rate,
                                        int shuffle,
                                        NN_Optimizer optimizer) {
    if (!model) return -1;
    model->impl.meta_has_locked = 1;
    model->impl.meta_locked_batch_size = batch_size;
    model->impl.meta_locked_learning_rate = learning_rate;
    model->impl.meta_locked_shuffle = shuffle ? 1 : 0;
    model->impl.meta_locked_optimizer = optimizer;
    return 0;
}

int nn_model_get_locked_training_params(const NN_Model* model,
                                        int* out_batch_size,
                                        float* out_learning_rate,
                                        int* out_shuffle,
                                        NN_Optimizer* out_optimizer) {
    if (!model) return -1;
    if (!model->impl.meta_has_locked) return -1;
    if (out_batch_size) *out_batch_size = model->impl.meta_locked_batch_size;
    if (out_learning_rate) *out_learning_rate = model->impl.meta_locked_learning_rate;
    if (out_shuffle) *out_shuffle = model->impl.meta_locked_shuffle;
    if (out_optimizer) *out_optimizer = model->impl.meta_locked_optimizer;
    return 0;
}

int nn_num_weight_layers(const NN_Model* model) {
    if (!model) return -1;
    return model->impl.num_weight_sets;
}

int nn_layer_dims(const NN_Model* model, int layer_index, int* out_in, int* out_out) {
    if (!model || layer_index < 0 || layer_index >= model->impl.num_weight_sets) return -1;
    int in = (layer_index == 0) ? model->impl.arch.input_neurons
                                : model->impl.arch.hidden_neurons_per_layer[layer_index - 1];
    int out = (layer_index < model->impl.arch.num_hidden_layers) ? model->impl.arch.hidden_neurons_per_layer[layer_index]
                                                                 : model->impl.arch.output_neurons;
    if (out_in) *out_in = in; if (out_out) *out_out = out; return 0;
}

int nn_get_weights(const NN_Model* model, int layer_index, float* out, size_t out_len) {
    if (!model || !out) return -1;
    int in=0, outn=0; if (nn_layer_dims(model, layer_index, &in, &outn) != 0) return -1;
    size_t need = (size_t)in * (size_t)outn;
    if (out_len < need) return -1;
    const float* W = model->impl.weights[layer_index];
    memcpy(out, W, need * sizeof(float));
    return 0;
}

int nn_get_biases(const NN_Model* model, int layer_index, float* out, size_t out_len) {
    if (!model || !out) return -1;
    int in=0, outn=0; if (nn_layer_dims(model, layer_index, &in, &outn) != 0) return -1;
    if (out_len < (size_t)outn) return -1;
    const float* B = model->impl.biases[layer_index];
    memcpy(out, B, (size_t)outn * sizeof(float));
    return 0;
}

int nn_compute_activations_from_image(const NN_Model* model, const char* image_path, int layer_index, float* out, size_t out_len) {
    if (!model || !image_path || !out) return -1;
    int L = model->impl.num_weight_sets;
    if (layer_index < 0 || layer_index > L) return -1;
    // Determine expected length
    int expected_len = (layer_index == 0) ? model->impl.arch.input_neurons
                        : (layer_index < model->impl.arch.num_hidden_layers ? model->impl.arch.hidden_neurons_per_layer[layer_index]
                                                                           : model->impl.arch.output_neurons);
    if (out_len < (size_t)expected_len) return -1;
    float* x = (float*)malloc(sizeof(float) * EXPECTED_INPUT_NEURONS);
    if (!x) return -1;
    if (load_and_process_image(image_path, x) != 0) { free(x); return -1; }
    if (layer_index == 0) {
        memcpy(out, x, (size_t)expected_len * sizeof(float));
        free(x); return 0;
    }
    // Forward until target layer
    int prev = model->impl.arch.input_neurons;
    float* a_prev = x; // owned
    for (int i=0;i<layer_index;i++) {
        int curr = (i < model->impl.arch.num_hidden_layers) ? model->impl.arch.hidden_neurons_per_layer[i]
                                                            : model->impl.arch.output_neurons;
        float* a_curr = (float*)malloc(sizeof(float) * curr);
        if (!a_curr) { free(a_prev); return -1; }
        const float* Wi = model->impl.weights[i]; const float* bi = model->impl.biases[i];
        for (int r=0;r<curr;++r) {
            double sum = bi[r]; const float* wrow = Wi + (long)r * prev; for (int c=0;c<prev;++c) sum += wrow[c]*a_prev[c];
            a_curr[r] = model->impl.layers[i].activate((float)sum);
        }
        if (i == layer_index - 1) {
            memcpy(out, a_curr, (size_t)expected_len * sizeof(float));
            free(a_curr); free(a_prev); return 0;
        }
        free(a_prev); a_prev = a_curr; prev = curr;
    }
    free(a_prev);
    return -1;
}

int nn_compute_pre_activations_from_image(const NN_Model* model, const char* image_path, int layer_index, float* out, size_t out_len) {
    if (!model || !image_path || !out) return -1;
    int L = model->impl.num_weight_sets;
    if (layer_index < 1 || layer_index > L) return -1;
    int expected_len = (layer_index < model->impl.arch.num_hidden_layers) ? model->impl.arch.hidden_neurons_per_layer[layer_index]
                                                                          : model->impl.arch.output_neurons;
    // Correction: for z at layer_index, size equals that layer's neuron count
    expected_len = (layer_index <= model->impl.arch.num_hidden_layers) ? model->impl.arch.hidden_neurons_per_layer[layer_index-1]
                                                                      : model->impl.arch.output_neurons;
    if (out_len < (size_t)expected_len) return -1;
    float* x = (float*)malloc(sizeof(float) * EXPECTED_INPUT_NEURONS);
    if (!x) return -1;
    if (load_and_process_image(image_path, x) != 0) { free(x); return -1; }
    int prev = model->impl.arch.input_neurons;
    float* a_prev = x; // owned
    for (int i=0;i<layer_index;i++) {
        int curr = (i < model->impl.arch.num_hidden_layers) ? model->impl.arch.hidden_neurons_per_layer[i]
                                                            : model->impl.arch.output_neurons;
        float* z_curr = (float*)malloc(sizeof(float) * curr);
        if (!z_curr) { free(a_prev); return -1; }
        const float* Wi = model->impl.weights[i]; const float* bi = model->impl.biases[i];
        for (int r=0;r<curr;++r) {
            double sum = bi[r]; const float* wrow = Wi + (long)r * prev; for (int c=0;c<prev;++c) sum += wrow[c]*a_prev[c];
            z_curr[r] = (float)sum;
        }
        if (i == layer_index - 1) {
            memcpy(out, z_curr, (size_t)expected_len * sizeof(float));
            free(z_curr); free(a_prev); return 0;
        }
        // Prepare next a_prev by applying activation to z
        float* a_curr = (float*)malloc(sizeof(float) * curr);
        if (!a_curr) { free(z_curr); free(a_prev); return -1; }
        for (int r=0;r<curr;++r) a_curr[r] = model->impl.layers[i].activate(z_curr[r]);
        free(z_curr);
        free(a_prev); a_prev = a_curr; prev = curr;
    }
    free(a_prev);
    return -1;
}
int nn_train_from_dirs(NN_Model* model,
                       const char* pos_dir,
                       const char* neg_dir,
                       const char* val_dir,
                       int steps,
                       int batch_size,
                       int shuffle,
                       float learning_rate,
                       float l1_lambda,
                       float l2_lambda,
                       float* out_last_loss,
                       float* out_val_loss) {
    if (!model || !pos_dir || steps <= 0 || batch_size <= 0) return -1;
    char** pos_list=NULL; int pos_count=0;
    char** neg_list=NULL; int neg_count=0;
    if (list_images_in_dir(pos_dir, &pos_list, &pos_count) != 0 || pos_count == 0) { free_string_array(pos_list, pos_count); return -1; }
    if (neg_dir && list_images_in_dir(neg_dir, &neg_list, &neg_count) != 0) { free_string_array(pos_list, pos_count); free_string_array(neg_list, neg_count); return -1; }
    int total = pos_count + neg_count;
    float* x = (float*)malloc(sizeof(float) * EXPECTED_INPUT_NEURONS);
    if (!x) { free_string_array(pos_list, pos_count); free_string_array(neg_list, neg_count); return -1; }
    float last_loss = 0.0f;
    int idx = 0;
    // Combined indices for optional shuffling
    int combined_count = total;
    int* order = NULL;
    int* labels = NULL; // 1=pos, 0=neg
    if (shuffle && combined_count > 0) {
        order = (int*)malloc(sizeof(int) * (size_t)combined_count);
        labels = (int*)malloc(sizeof(int) * (size_t)combined_count);
        if (!order || !labels) { free(order); free(labels); free(x); free_string_array(pos_list,pos_count); free_string_array(neg_list,neg_count); return -1; }
        int k=0; for (int i=0;i<pos_count;++i) { order[k]=i; labels[k]=1; k++; }
        for (int j=0;j<neg_count;++j) { order[k]=j; labels[k]=0; k++; }
        srand((unsigned int)time(NULL));
        for (int i=combined_count-1; i>0; --i) { int j = rand() % (i+1); int tmp=order[i]; order[i]=order[j]; order[j]=tmp; int lt=labels[i]; labels[i]=labels[j]; labels[j]=lt; }
    }
    // Optional validation sets: expect val_dir/pos and val_dir/neg
    char **val_pos_list=NULL, **val_neg_list=NULL; int val_pos_count=0, val_neg_count=0;
    if (val_dir) {
        char path_pos[4096]; char path_neg[4096];
        snprintf(path_pos,sizeof(path_pos),"%s/pos", val_dir);
        snprintf(path_neg,sizeof(path_neg),"%s/neg", val_dir);
        if (list_images_in_dir(path_pos, &val_pos_list, &val_pos_count) != 0) { val_pos_list=NULL; val_pos_count=0; }
        if (list_images_in_dir(path_neg, &val_neg_list, &val_neg_count) != 0) { val_neg_list=NULL; val_neg_count=0; }
        // If neither subdir exists or empty, fall back to treating val_dir as positives-only
        if (val_pos_count==0 && val_neg_count==0) {
            (void)list_images_in_dir(val_dir, &val_pos_list, &val_pos_count);
        }
    }
    for (int step = 0; step < steps; ++step) {
        // reshuffle at each step/epoch if requested
        if (shuffle && order) {
            for (int i=combined_count-1; i>0; --i) { int j = rand() % (i+1); int tmp=order[i]; order[i]=order[j]; order[j]=tmp; int lt=labels[i]; labels[i]=labels[j]; labels[j]=lt; }
            idx = 0;
        }
        float batch_loss = 0.0f; int batch_items = 0;
        for (int b = 0; b < batch_size; ++b) {
            const char* path = NULL; float y = 0.0f;
            if (shuffle && order) {
                if (idx >= combined_count) idx = 0;
                int is_pos = labels[idx];
                if (is_pos) { int pidx = order[idx]; if (pidx >= pos_count) pidx %= (pos_count>0?pos_count:1); path = pos_list[pidx]; y = 1.0f; }
                else { int nidx = order[idx]; if (nidx >= neg_count) nidx %= (neg_count>0?neg_count:1); if (neg_count>0) { path = neg_list[nidx]; y = 0.0f; } else { path = pos_list[nidx % (pos_count>0?pos_count:1)]; y = 0.0f; } }
                idx++;
            } else {
                // Round-robin through positives then negatives
                int use_pos = (idx % (total)) < pos_count;
                if (use_pos) { path = pos_list[idx % pos_count]; y = 1.0f; }
                else { int j = (idx - pos_count) % (neg_count > 0 ? neg_count : 1); if (neg_count > 0) path = neg_list[j]; else path = pos_list[idx % pos_count]; y = 0.0f; }
                idx++;
            }
            if (load_and_process_image(path, x) != 0) continue;
            float loss = train_one_example(&model->impl, x, y, learning_rate, l1_lambda, l2_lambda);
            if (!isnan(loss) && !isinf(loss)) { batch_loss += loss; batch_items++; }
        }
        if (batch_items > 0) last_loss = batch_loss / batch_items;
        // Per-epoch validation loss (optional)
        if (val_dir && out_val_loss) {
            float vloss_sum = 0.0f; int vcount = 0;
            // Evaluate positives
            float* vx = (float*)malloc(sizeof(float) * EXPECTED_INPUT_NEURONS);
            if (vx) {
                int max_samples = 128;
                if (val_pos_count > 0) {
                    int stride = (val_pos_count > max_samples) ? (val_pos_count / max_samples) : 1;
                    for (int i=0;i<val_pos_count;i+=stride) {
                        if (load_and_process_image(val_pos_list[i], vx) != 0) continue;
                        float l = compute_loss_only(&model->impl, vx, 1.0f);
                        if (!isnan(l) && !isinf(l)) { vloss_sum += l; vcount++; }
                        if (vcount >= max_samples) break;
                    }
                }
                if (val_neg_count > 0) {
                    int stride = (val_neg_count > max_samples) ? (val_neg_count / max_samples) : 1;
                    for (int i=0;i<val_neg_count;i+=stride) {
                        if (load_and_process_image(val_neg_list[i], vx) != 0) continue;
                        float l = compute_loss_only(&model->impl, vx, 0.0f);
                        if (!isnan(l) && !isinf(l)) { vloss_sum += l; vcount++; }
                        if (vcount >= 2*max_samples) break;
                    }
                }
                // If only a flat val dir with unknown labels, treat as positives
                if (val_pos_count>0 && val_neg_count==0) {
                    // already handled above as positives
                }
                free(vx);
            }
            if (vcount > 0) *out_val_loss = vloss_sum / vcount;
        }
    }
    free(x);
    if (order) free(order);
    if (labels) free(labels);
    free_string_array(pos_list, pos_count);
    free_string_array(neg_list, neg_count);
    free_string_array(val_pos_list, val_pos_count);
    free_string_array(val_neg_list, val_neg_count);
    if (out_last_loss) *out_last_loss = last_loss;
    return 0;
}

// Down/upsample a row of length src_len into dst_len using simple bin-averaging (down) or nearest (up)
static void resample_row_uniform(const float* src, int src_len, float* dst, int dst_len) {
    if (dst_len <= 0 || src_len <= 0) return;
    if (src_len == dst_len) { memcpy(dst, src, sizeof(float) * (size_t)dst_len); return; }
    if (src_len > dst_len) {
        for (int j = 0; j < dst_len; ++j) {
            long start = ((long)j * (long)src_len) / (long)dst_len;
            long end   = ((long)(j + 1) * (long)src_len) / (long)dst_len;
            if (end <= start) end = start + 1;
            if (end > src_len) end = src_len;
            double sum = 0.0; long cnt = 0;
            for (long k = start; k < end; ++k) { sum += src[k]; cnt++; }
            dst[j] = (cnt > 0) ? (float)(sum / (double)cnt) : 0.0f;
        }
    } else { // upsample: nearest neighbor
        for (int j = 0; j < dst_len; ++j) {
            long idx = ((long)j * (long)src_len) / (long)dst_len;
            if (idx >= src_len) idx = src_len - 1;
            dst[j] = src[idx];
        }
    }
}

// Mapping helpers
static void map_to_bytes_minmax(const float* in, unsigned char* out, long n, float vmin, float vmax) {
    if (vmax > vmin) {
        double scale = 255.0 / (double)(vmax - vmin);
        double bias  = - (double)vmin * scale;
        for (long t = 0; t < n; ++t) {
            int iv = (int)((double)in[t] * scale + bias + 0.5);
            if (iv < 0) iv = 0; if (iv > 255) iv = 255; out[t] = (unsigned char)iv;
        }
    } else {
        memset(out, 127, (size_t)n);
    }
}

static void map_to_bytes_symmetric_zero(const float* in, unsigned char* out, long n) {
    float maxabs = 0.0f;
    for (long t = 0; t < n; ++t) { float a = fabsf(in[t]); if (a > maxabs) maxabs = a; }
    if (maxabs > 0.0f) {
        double scale = 127.5 / (double)maxabs; // [-max,+max] -> [-127.5,+127.5]
        for (long t = 0; t < n; ++t) {
            int iv = (int)(in[t] * scale + 127.5 + 0.5);
            if (iv < 0) iv = 0; if (iv > 255) iv = 255; out[t] = (unsigned char)iv;
        }
    } else {
        memset(out, 127, (size_t)n);
    }
}

// Render a single hidden layer visualization to an 8-bit grayscale buffer.
// The caller is responsible for free() on out_pixels.
int nn_render_hidden_layer_visualization(const NN_Model* model,
                                         int layer_index,
                                         const NN_VisOptions* options,
                                         unsigned char** out_pixels,
                                         int* out_width,
                                         int* out_height) {
    if (!model || !out_pixels || !out_width || !out_height) return -1;
    if (layer_index < 0 || layer_index >= model->impl.arch.num_hidden_layers) return -1;
    NN_VisMode mode = NN_VIS_MODE_WEIGHTS;
    NN_VisScale scale = NN_VIS_SCALE_MINMAX;
    int raw_weights_full = 0;
    if (options) { mode = options->mode; scale = options->scale; raw_weights_full = options->raw_weights_full; }

    int prev = model->impl.arch.input_neurons;
    for (int i=0;i<layer_index;i++) prev = model->impl.arch.hidden_neurons_per_layer[i];
    int curr = model->impl.arch.hidden_neurons_per_layer[layer_index];
    const float* Wi = model->impl.weights[layer_index];

    unsigned char* img = NULL; float* work = NULL; long workN = 0; float minv=0.0f, maxv=0.0f;
    if (mode == NN_VIS_MODE_WEIGHTS) {
        if (raw_weights_full) {
            *out_width = prev; *out_height = curr; workN = (long)curr * (long)prev;
            work = (float*)malloc(sizeof(float) * (size_t)workN);
            if (!work) return -1;
            for (int r=0;r<curr;++r) {
                const float* wrow = Wi + (long)r * prev;
                memcpy(work + (size_t)r * (size_t)prev, wrow, sizeof(float) * (size_t)prev);
            }
        } else {
            *out_width = curr; *out_height = curr; workN = (long)curr * (long)curr;
            work = (float*)malloc(sizeof(float) * (size_t)workN);
            if (!work) return -1;
            for (int r=0;r<curr;++r) {
                const float* wrow = Wi + (long)r * prev;
                resample_row_uniform(wrow, prev, work + (size_t)r * (size_t)curr, curr);
            }
        }
        minv = maxv = work[0];
        for (long t=1;t<workN;++t) { float v=work[t]; if (v<minv) minv=v; if (v>maxv) maxv=v; }
        img = (unsigned char*)malloc((size_t)workN);
        if (!img) { free(work); return -1; }
        if (scale == NN_VIS_SCALE_MINMAX) map_to_bytes_minmax(work, img, workN, minv, maxv);
        else map_to_bytes_symmetric_zero(work, img, workN);
        free(work);
    } else { // heatmap
        const int max_samples = 1024;
        int use_exact = (prev <= 8192);
        int samples = use_exact ? prev : ((prev < max_samples) ? prev : max_samples);
        int* idx = NULL;
        if (!use_exact) {
            idx = (int*)malloc(sizeof(int) * (size_t)samples);
            if (!idx) return -1;
            for (int k=0;k<samples;++k) {
                long pos = ((long)k * (long)prev) / (long)samples;
                if (pos >= prev) pos = prev - 1;
                idx[k] = (int)pos;
            }
        }
        float* proj = use_exact ? NULL : (float*)malloc(sizeof(float) * (size_t)curr * (size_t)samples);
        double* norms = (double*)malloc(sizeof(double) * (size_t)curr);
        if (!norms || (!use_exact && !proj)) { free(idx); free(proj); free(norms); return -1; }
        for (int r=0;r<curr;++r) {
            const float* wr = Wi + (long)r * prev; double nrm = 0.0;
            if (use_exact) {
                for (int k=0;k<samples;++k) { float v = wr[k]; nrm += (double)v*v; }
            } else {
                float* rowp = proj + (size_t)r * (size_t)samples;
                for (int k=0;k<samples;++k) { float v = wr[idx[k]]; rowp[k] = v; nrm += (double)v*v; }
            }
            norms[r] = sqrt(nrm);
        }
        long N = (long)curr * (long)curr; work = (float*)malloc(sizeof(float) * (size_t)N);
        if (!work) { free(idx); free(proj); free(norms); return -1; }
        for (long t=0;t<N;++t) work[t]=0.0f; float minvv=1.0f, maxvv=-1.0f;
        for (int r=0;r<curr;++r) {
            work[(size_t)r * curr + r] = 1.0f;
            const float* pr = use_exact ? (Wi + (long)r * prev) : (proj + (size_t)r * (size_t)samples); double nr = norms[r];
            for (int c=r+1;c<curr;++c) {
                const float* pc = use_exact ? (Wi + (long)c * prev) : (proj + (size_t)c * (size_t)samples); double nc = norms[c]; double dot=0.0;
                for (int k=0;k<samples;++k) dot += (double)pr[k] * (double)pc[k];
                float cs=0.0f; if (nr>0.0 && nc>0.0) { double v = dot/(nr*nc); if (v<-1.0) v=-1.0; if (v>1.0) v=1.0; cs=(float)v; }
                work[(size_t)r * curr + c] = cs; work[(size_t)c * curr + r] = cs; if (cs<minvv && r!=c) minvv=cs; if (cs>maxvv && r!=c) maxvv=cs;
            }
        }
        if (idx) free(idx); if (proj) free(proj); free(norms);
        img = (unsigned char*)malloc((size_t)N); if (!img){ free(work); return -1; }
        if (scale == NN_VIS_SCALE_MINMAX) { if (maxvv>minvv) map_to_bytes_minmax(work, img, N, minvv, maxvv); else map_to_bytes_symmetric_zero(work, img, N); }
        else map_to_bytes_symmetric_zero(work, img, N);
        *out_width = curr; *out_height = curr; free(work);
    }
    *out_pixels = img; return 0;
}

int nn_export_layer_visualizations_ex(const NN_Model* model, const char* output_dir, const NN_VisOptions* options) {
    if (!model || !output_dir) return -1;
    struct stat st; if (stat(output_dir, &st) != 0) { (void)mkdir(output_dir, 0755); }
    NN_VisMode mode = NN_VIS_MODE_WEIGHTS;
    NN_VisScale scale = NN_VIS_SCALE_MINMAX;
    int include_bias = 0, include_stats = 0, raw_weights_full = 0;
    int only_layer = -1;
    if (options) { mode = options->mode; scale = options->scale; include_bias = options->include_bias; include_stats = options->include_stats; raw_weights_full = options->raw_weights_full; only_layer = options->only_layer; }
    int prev = model->impl.arch.input_neurons;
    for (int i = 0; i < model->impl.arch.num_hidden_layers; ++i) {
        if (only_layer >= 0 && i != only_layer) { prev = model->impl.arch.hidden_neurons_per_layer[i]; continue; }
        int curr = model->impl.arch.hidden_neurons_per_layer[i];
        const float* Wi = model->impl.weights[i];
        unsigned char* img = NULL;
        float* work = NULL;
        long workN = 0;
        float minv=0.0f, maxv=0.0f;

        if (mode == NN_VIS_MODE_WEIGHTS) {
            if (raw_weights_full) {
                // Export raw weight matrix as non-square PGM: width=prev, height=curr
                workN = (long)curr * (long)prev;
                work = (float*)malloc(sizeof(float) * (size_t)workN);
                if (!work) return -1;
                for (int r=0; r<curr; ++r) {
                    const float* wrow = Wi + (long)r * prev;
                    memcpy(work + (size_t)r * (size_t)prev, wrow, sizeof(float) * (size_t)prev);
                }
                minv = maxv = work[0];
                for (long t=1; t<workN; ++t) { float v=work[t]; if(v<minv) minv=v; if(v>maxv) maxv=v; }
                img = (unsigned char*)malloc((size_t)workN);
                if (!img) { free(work); return -1; }
                if (scale == NN_VIS_SCALE_MINMAX) map_to_bytes_minmax(work, img, workN, minv, maxv);
                else map_to_bytes_symmetric_zero(work, img, workN);
                char path_raw[4096]; snprintf(path_raw,sizeof(path_raw),"%s/layer_%d_raw.pgm", output_dir, i+1);
                int rc_raw = write_pgm(path_raw, prev, curr, img);
                free(img);
                if (include_stats) {
                    double sum=0.0, sum2=0.0; for (long t=0;t<workN;++t){ double v=work[t]; sum+=v; sum2+=v*v; } double mean=sum/(double)workN; double var=(sum2/(double)workN)-mean*mean; if(var<0)var=0; double std=sqrt(var);
                    char spath[4096]; snprintf(spath,sizeof(spath),"%s/layer_%d_raw_stats.txt", output_dir, i+1);
                    FILE* sf=fopen(spath,"w"); if (sf){ fprintf(sf,"min=%.9g\nmax=%.9g\nmean=%.9g\nstd=%.9g\n", (double)minv, (double)maxv, mean, std); fclose(sf);} }
                free(work);
                if (rc_raw != 0) return -1;
                // Continue to next layer without producing square image when raw requested
                prev = curr;
                // Also export biases if requested (handled below after work was freed; replicate minimal code)
                if (include_bias) {
                    const float* b = model->impl.biases[i];
                    float bmin=b[0], bmax=b[0]; for (int j=1;j<curr;++j){ float v=b[j]; if(v<bmin) bmin=v; if(v>bmax) bmax=v; }
                    unsigned char* brow = (unsigned char*)malloc((size_t)curr);
                    if (brow) {
                        if (scale == NN_VIS_SCALE_MINMAX) map_to_bytes_minmax(b, brow, curr, bmin, bmax);
                        else map_to_bytes_symmetric_zero(b, brow, curr);
                        char bpath[4096]; snprintf(bpath,sizeof(bpath),"%s/layer_%d_biases.pgm", output_dir, i+1);
                        (void)write_pgm(bpath, curr, 1, brow);
                        free(brow);
                        if (include_stats) {
                            double s=0.0,s2=0.0; for (int j=0;j<curr;++j){ double v=b[j]; s+=v; s2+=v*v; } double mean=s/(double)curr; double var=(s2/(double)curr)-mean*mean; if (var<0)var=0; double std=sqrt(var);
                            char bsp[4096]; snprintf(bsp,sizeof(bsp),"%s/layer_%d_biases_stats.txt", output_dir, i+1);
                            FILE* sf=fopen(bsp,"w"); if (sf){ fprintf(sf,"min=%.9g\nmax=%.9g\nmean=%.9g\nstd=%.9g\n", (double)bmin,(double)bmax,mean,std); fclose(sf);} }
                    }
                }
                continue;
            }
            workN = (long)curr * (long)curr;
            work = (float*)malloc(sizeof(float) * (size_t)workN);
            if (!work) return -1;
            for (int r = 0; r < curr; ++r) {
                const float* wrow = Wi + (long)r * prev;
                resample_row_uniform(wrow, prev, work + (size_t)r * (size_t)curr, curr);
            }
            minv = maxv = work[0];
            for (long t = 1; t < workN; ++t) { float v=work[t]; if (v<minv) minv=v; if (v>maxv) maxv=v; }
            img = (unsigned char*)malloc((size_t)workN);
            if (!img) { free(work); return -1; }
            if (scale == NN_VIS_SCALE_MINMAX) map_to_bytes_minmax(work, img, workN, minv, maxv);
            else map_to_bytes_symmetric_zero(work, img, workN);
        } else { // heatmap
            const int max_samples = 1024;
            int use_exact = (prev <= 8192);
            int samples = use_exact ? prev : ((prev < max_samples) ? prev : max_samples);
            int* idx = NULL;
            if (!use_exact) {
                idx = (int*)malloc(sizeof(int) * (size_t)samples);
                if (!idx) return -1;
                for (int k = 0; k < samples; ++k) {
                    long pos = ((long)k * (long)prev) / (long)samples;
                    if (pos >= prev) pos = prev - 1; idx[k] = (int)pos;
                }
            }
            float* proj = use_exact ? NULL : (float*)malloc(sizeof(float) * (size_t)curr * (size_t)samples);
            double* norms = (double*)malloc(sizeof(double) * (size_t)curr);
            if (!norms || (!use_exact && !proj)) { free(idx); free(proj); free(norms); return -1; }
            for (int r = 0; r < curr; ++r) {
                const float* wr = Wi + (long)r * prev;
                double nrm = 0.0;
                if (use_exact) {
                    for (int k = 0; k < samples; ++k) { float v = wr[k]; nrm += (double)v * v; }
                } else {
                    float* rowp = proj + (size_t)r * (size_t)samples;
                    for (int k = 0; k < samples; ++k) { float v = wr[idx[k]]; rowp[k] = v; nrm += (double)v * v; }
                }
                norms[r] = sqrt(nrm);
            }
            workN = (long)curr * (long)curr;
            work = (float*)malloc(sizeof(float) * (size_t)workN);
            if (!work) { free(idx); free(proj); free(norms); return -1; }
            for (long t=0;t<workN;++t) work[t]=0.0f;
            minv=1.0f; maxv=-1.0f;
            for (int r = 0; r < curr; ++r) {
                work[(size_t)r * curr + r] = 1.0f;
                const float* pr = use_exact ? (Wi + (long)r * prev) : (proj + (size_t)r * (size_t)samples);
                double nr = norms[r];
                for (int c = r+1; c < curr; ++c) {
                    const float* pc = use_exact ? (Wi + (long)c * prev) : (proj + (size_t)c * (size_t)samples);
                    double nc = norms[c]; double dot=0.0;
                    for (int k=0;k<samples;++k) dot += (double)pr[k] * (double)pc[k];
                    float cs = 0.0f;
                    if (nr>0.0 && nc>0.0) { double v = dot/(nr*nc); if (v<-1.0) v=-1.0; if (v>1.0) v=1.0; cs=(float)v; }
                    work[(size_t)r * curr + c] = cs; work[(size_t)c * curr + r] = cs;
                    if (cs<minv && r!=c) minv=cs; if (cs>maxv && r!=c) maxv=cs;
                }
            }
            if (idx) free(idx); if (proj) free(proj); free(norms);
            img = (unsigned char*)malloc((size_t)workN);
            if (!img) { free(work); return -1; }
            if (scale == NN_VIS_SCALE_MINMAX) {
                if (maxv > minv) map_to_bytes_minmax(work, img, workN, minv, maxv);
                else map_to_bytes_symmetric_zero(work, img, workN);
            } else {
                map_to_bytes_symmetric_zero(work, img, workN);
            }
            for (int d=0; d<curr; ++d) img[(size_t)d * curr + d] = 255;
        }

        char path[4096];
        snprintf(path, sizeof(path), "%s/layer_%d.pgm", output_dir, i+1);
        int rc = write_pgm(path, curr, curr, img);
        free(img);
        if (include_stats) {
            double sum=0.0, sum2=0.0; for (long t=0;t<workN;++t){ double v=work[t]; sum+=v; sum2+=v*v; }
            double mean = sum/(double)workN; double var = (sum2/(double)workN) - mean*mean; if (var<0) var=0; double std = sqrt(var);
            char spath[4096]; snprintf(spath,sizeof(spath),"%s/layer_%d_stats.txt", output_dir, i+1);
            FILE* sf = fopen(spath,"w"); if (sf){ fprintf(sf,"min=%.9g\nmax=%.9g\nmean=%.9g\nstd=%.9g\n", (double)minv, (double)maxv, mean, std); fclose(sf);} }
        free(work);
        if (include_bias) {
            const float* b = model->impl.biases[i];
            float bmin=b[0], bmax=b[0]; for (int j=1;j<curr;++j){ float v=b[j]; if(v<bmin) bmin=v; if(v>bmax) bmax=v; }
            unsigned char* brow = (unsigned char*)malloc((size_t)curr);
            if (brow) {
                if (scale == NN_VIS_SCALE_MINMAX) map_to_bytes_minmax(b, brow, curr, bmin, bmax);
                else map_to_bytes_symmetric_zero(b, brow, curr);
                char bpath[4096]; snprintf(bpath,sizeof(bpath),"%s/layer_%d_biases.pgm", output_dir, i+1);
                (void)write_pgm(bpath, curr, 1, brow);
                free(brow);
                if (include_stats) {
                    double s=0.0,s2=0.0; for (int j=0;j<curr;++j){ double v=b[j]; s+=v; s2+=v*v; } double mean=s/(double)curr; double var=(s2/(double)curr)-mean*mean; if (var<0)var=0; double std=sqrt(var);
                    char bsp[4096]; snprintf(bsp,sizeof(bsp),"%s/layer_%d_biases_stats.txt", output_dir, i+1);
                    FILE* sf=fopen(bsp,"w"); if (sf){ fprintf(sf,"min=%.9g\nmax=%.9g\nmean=%.9g\nstd=%.9g\n", (double)bmin,(double)bmax,mean,std); fclose(sf);} }
            }
        }
        if (rc != 0) return -1;
        prev = curr;
    }
    return 0;
}

int nn_export_layer_visualizations(const NN_Model* model, const char* output_dir) {
    NN_VisOptions opts; opts.mode = NN_VIS_MODE_WEIGHTS; opts.scale = NN_VIS_SCALE_MINMAX; opts.include_bias = 0; opts.include_stats = 0;
    return nn_export_layer_visualizations_ex(model, output_dir, &opts);
}

// replaced by nn_export_layer_visualizations_ex wrapper below

int nn_evaluate_dirs(const NN_Model* model,
                     const char* pos_dir,
                     const char* neg_dir,
                     int* true_pos,
                     int* true_neg,
                     int* false_pos,
                     int* false_neg,
                     float* out_accuracy) {
    if (!model) return -1;
    int tp=0, tn=0, fp=0, fn=0;
    char** pos_list=NULL; int pos_count=0;
    char** neg_list=NULL; int neg_count=0;
    if (pos_dir && list_images_in_dir(pos_dir,&pos_list,&pos_count)!=0) { pos_list=NULL; pos_count=0; }
    if (neg_dir && list_images_in_dir(neg_dir,&neg_list,&neg_count)!=0) { neg_list=NULL; neg_count=0; }
    float* x = (float*)malloc(sizeof(float) * EXPECTED_INPUT_NEURONS);
    if (!x) { free_string_array(pos_list,pos_count); free_string_array(neg_list,neg_count); return -1; }
    for (int i=0;i<pos_count;++i) {
        if (load_and_process_image(pos_list[i], x) != 0) continue;
        float p = perform_forward_pass(&model->impl, x);
        if (p>0.5f) tp++; else fn++;
    }
    for (int i=0;i<neg_count;++i) {
        if (load_and_process_image(neg_list[i], x) != 0) continue;
        float p = perform_forward_pass(&model->impl, x);
        if (p>0.5f) fp++; else tn++;
    }
    free(x);
    free_string_array(pos_list,pos_count);
    free_string_array(neg_list,neg_count);
    int total = tp+tn+fp+fn;
    if (true_pos) *true_pos = tp;
    if (true_neg) *true_neg = tn;
    if (false_pos) *false_pos = fp;
    if (false_neg) *false_neg = fn;
    if (out_accuracy) *out_accuracy = (total>0)? ((float)(tp+tn)/(float)total) : 0.0f;
    return 0;
}

static float perform_forward_pass(const MLPModel* model, const float* input_data) {
    float* current_activations = (float*)input_data; // Initially points to input_data
    float* next_activations_buffer = NULL; // To store activations of the next layer
    int prev_layer_num_neurons = model->arch.input_neurons;

    // Iterate through all layers (input -> hidden1 ... -> output)
    for (int layer_idx = 0; layer_idx < model->num_weight_sets; ++layer_idx) {
        int current_layer_num_neurons = (layer_idx < model->arch.num_hidden_layers)
            ? model->arch.hidden_neurons_per_layer[layer_idx]
            : model->arch.output_neurons;

        next_activations_buffer = (float*)malloc((size_t)current_layer_num_neurons * sizeof(float));
        if (!next_activations_buffer) {
            fprintf(stderr, "NoodleNet Error: Malloc failed for activations buffer.\n");
            if (current_activations != input_data) free(current_activations);
            return NAN;
        }

        const float* layer_weights = model->weights[layer_idx];
        const float* layer_biases = model->biases[layer_idx];

#if NN_USE_ACCELERATE
        // next_activations_buffer = act( W * current_activations + b )
        // compute z into temp buffer
        float* zbuf = (float*)malloc((size_t)current_layer_num_neurons * sizeof(float));
        if (!zbuf) {
            if (current_activations != input_data) free(current_activations);
            free(next_activations_buffer);
            return NAN;
        }
        memcpy(zbuf, layer_biases, sizeof(float) * (size_t)current_layer_num_neurons);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    current_layer_num_neurons, prev_layer_num_neurons,
                    1.0f, layer_weights, prev_layer_num_neurons,
                    current_activations, 1,
                    1.0f, zbuf, 1);
        for (int j = 0; j < current_layer_num_neurons; ++j) {
            next_activations_buffer[j] = model->layers[layer_idx].activate(zbuf[j]);
        }
        free(zbuf);
#else
        for (int j = 0; j < current_layer_num_neurons; ++j) {
            double sum = layer_biases[j];
            const float* wrow = layer_weights + (long)j * prev_layer_num_neurons;
            for (int i = 0; i < prev_layer_num_neurons; ++i) sum += wrow[i] * current_activations[i];
            next_activations_buffer[j] = model->layers[layer_idx].activate((float)sum);
        }
#endif

        if (current_activations != input_data) free(current_activations);
        current_activations = next_activations_buffer;
        next_activations_buffer = NULL;
        prev_layer_num_neurons = current_layer_num_neurons;
    }

    float final_prediction = current_activations[0];
    if (current_activations != input_data) free(current_activations);
    return final_prediction;
}

// ---- Optimizer configuration ----
int nn_model_set_optimizer(NN_Model* model, NN_Optimizer opt, float beta1, float beta2, float epsilon) {
    if (!model) return -1;
    model->impl.optimizer = opt;
    if (opt == NN_OPTIMIZER_RMSPROP || opt == NN_OPTIMIZER_ADAM) {
        if (beta1 <= 0.0f || beta1 >= 1.0f) beta1 = 0.9f;
        if (beta2 <= 0.0f || beta2 >= 1.0f) beta2 = 0.999f;
        if (epsilon <= 0.0f) epsilon = 1e-8f;
    }
    model->impl.beta1 = beta1;
    model->impl.beta2 = beta2;
    model->impl.epsilon = epsilon;
    // Reset timestep when switching optimizers
    model->impl.adam_timestep = 0;
    return 0;
}
