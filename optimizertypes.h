#ifndef OPTIMIZERTYPES_H
#define OPTIMIZERTYPES_H

/**
 * @brief Enumeration of available optimization algorithms
 */
enum class OptimizerType {
    SGD,        ///< Stochastic Gradient Descent
    RMSprop,    ///< Root Mean Square Propagation
    Adam        ///< Adaptive Moment Estimation
};

#endif // OPTIMIZERTYPES_H
