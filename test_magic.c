#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define MODEL_MAGIC_NUMBER 0x4D4E4553 // "SENM" in little-endian (S E N M)
#define MODEL_MAGIC_NUMBER_REVERSED 0x53454E4D // "SENM" in big-endian (M N E S)

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <model_file>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];

    // Open the model file
    FILE *model_file = fopen(model_path, "rb");
    if (!model_file) {
        fprintf(stderr, "Error: Could not open model file '%s'\n", model_path);
        return 1;
    }

    // Read the first 8 bytes
    unsigned char buffer[8];
    if (fread(buffer, 1, 8, model_file) != 8) {
        fprintf(stderr, "Error: Failed to read from '%s'\n", model_path);
        fclose(model_file);
        return 1;
    }

    // Print each byte
    printf("First 8 bytes (hex): ");
    for (int i = 0; i < 8; i++) {
        printf("%02X ", buffer[i]);
    }
    printf("\n");

    // Print as ASCII
    printf("First 8 bytes (ASCII): ");
    for (int i = 0; i < 8; i++) {
        if (buffer[i] >= 32 && buffer[i] <= 126) {
            printf("%c", buffer[i]);
        } else {
            printf(".");
        }
    }
    printf("\n");

    // Interpret as uint32_t in different ways
    uint32_t magic_number =
        (buffer[0] << 24) |
        (buffer[1] << 16) |
        (buffer[2] << 8) |
        buffer[3];

    uint32_t magic_number_reversed =
        (buffer[3] << 24) |
        (buffer[2] << 16) |
        (buffer[1] << 8) |
        buffer[0];

    printf("Magic number (big-endian interpretation): 0x%08X\n", magic_number);
    printf("Magic number (little-endian interpretation): 0x%08X\n", magic_number_reversed);

    // Direct comparison with expected values
    printf("Expected (little-endian): 0x%08X\n", MODEL_MAGIC_NUMBER);
    printf("Expected (big-endian): 0x%08X\n", MODEL_MAGIC_NUMBER_REVERSED);

    // Check if either interpretation matches
    if (magic_number == MODEL_MAGIC_NUMBER_REVERSED) {
        printf("Big-endian interpretation matches big-endian format (SENM)\n");
    } else if (magic_number_reversed == MODEL_MAGIC_NUMBER) {
        printf("Little-endian interpretation matches little-endian format (SENM)\n");
    } else {
        printf("Neither interpretation matches expected formats\n");
    }

    fclose(model_file);
    return 0;
}
