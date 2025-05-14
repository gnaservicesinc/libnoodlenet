#include <iostream>
#include <string>
#include <QApplication>
#include <QImage>
#include <QDebug>
#include <QFile>
#include <QDataStream>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include "mlp.h"
#include <noodlenet.hpp>

// This program compares predictions between sensuser's MLP and libnoodlenet
// for the same model and image

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
        return 1;
    }

    QApplication app(argc, argv);
    
    const char* model_path = argv[1];
    const char* image_path = argv[2];
    
    // Load the image
    QImage image(image_path);
    if (image.isNull()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    
    // Create MLP and load model
    MLP mlp(512 * 512, 128, 1, "sigmoid", "sigmoid");
    bool loaded = mlp.loadFromBinary(model_path);
    if (!loaded) {
        std::cerr << "Failed to load model in sensuser: " << model_path << std::endl;
        return 1;
    }
    
    // Get prediction from sensuser
    float sensuser_prediction = mlp.predict(image);
    bool sensuser_result = sensuser_prediction > 0.5f;
    
    // Get prediction from libnoodlenet
    int noodlenet_result = NoodleNet::predict(model_path, image_path);
    
    // Print results
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Sensuser prediction: " << sensuser_prediction 
              << " (" << (sensuser_result ? "positive" : "negative") << ")" << std::endl;
    std::cout << "Noodlenet result: " << noodlenet_result 
              << " (" << (noodlenet_result == 1 ? "positive" : "negative") << ")" << std::endl;
    
    // Compare preprocessing steps
    std::cout << "\nDebugging image preprocessing differences:" << std::endl;
    
    // Get preprocessed image from sensuser
    Eigen::VectorXf preprocessed = mlp.preprocessImage(image);
    
    // Save preprocessed image to a file for inspection
    QImage sensuser_processed(512, 512, QImage::Format_Grayscale8);
    for (int y = 0; y < 512; ++y) {
        for (int x = 0; x < 512; ++x) {
            int pixel_value = static_cast<int>(preprocessed(y * 512 + x) * 255.0f);
            sensuser_processed.setPixel(x, y, qRgb(pixel_value, pixel_value, pixel_value));
        }
    }
    sensuser_processed.save("sensuser_processed.png");
    
    std::cout << "Saved sensuser's preprocessed image to sensuser_processed.png" << std::endl;
    std::cout << "To compare with libnoodlenet, you would need to modify libnoodlenet to save its preprocessed image" << std::endl;
    
    return 0;
}
