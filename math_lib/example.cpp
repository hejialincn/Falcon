#include "include/mathlib.h"
#include <iostream>
#include <vector>

using namespace MathLib;

int main() {
    std::cout << "Math Library Example Program" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Basic arithmetic
    std::cout << "Basic Arithmetic:" << std::endl;
    std::cout << "5 + 3 = " << add(5, 3) << std::endl;
    std::cout << "5 - 3 = " << subtract(5, 3) << std::endl;
    std::cout << "5 * 3 = " << multiply(5, 3) << std::endl;
    std::cout << "5 / 3 = " << divide(5, 3) << std::endl;
    
    std::cout << std::endl;
    
    // Powers and roots
    std::cout << "Powers and Roots:" << std::endl;
    std::cout << "2^8 = " << power(2, 8) << std::endl;
    std::cout << "sqrt(16) = " << square_root(16) << std::endl;
    std::cout << "cbrt(27) = " << cube_root(27) << std::endl;
    
    std::cout << std::endl;
    
    // Trigonometry
    std::cout << "Trigonometry (in radians):" << std::endl;
    std::cout << "sin(PI/2) = " << sine(PI/2) << std::endl;
    std::cout << "cos(0) = " << cosine(0) << std::endl;
    std::cout << "tan(PI/4) = " << tangent(PI/4) << std::endl;
    
    std::cout << std::endl;
    
    // Logarithms
    std::cout << "Logarithms:" << std::endl;
    std::cout << "ln(e) = " << log_natural(E) << std::endl;
    std::cout << "log10(1000) = " << log_base10(1000) << std::endl;
    
    std::cout << std::endl;
    
    // Factorial
    std::cout << "Factorial:" << std::endl;
    std::cout << "5! = " << factorial(5) << std::endl;
    std::cout << "C(5,2) = " << combination(5, 2) << std::endl;
    std::cout << "P(5,2) = " << permutation(5, 2) << std::endl;
    
    std::cout << std::endl;
    
    // Statistics
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::cout << "Statistics on [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:" << std::endl;
    std::cout << "Mean = " << mean(data) << std::endl;
    std::cout << "Median = " << median(data) << std::endl;
    std::cout << "Standard Deviation = " << standard_deviation(data) << std::endl;
    std::cout << "Variance = " << variance(data) << std::endl;
    std::cout << "Max = " << max_value(data) << std::endl;
    std::cout << "Min = " << min_value(data) << std::endl;
    
    std::cout << std::endl;
    
    // Matrices
    std::vector<std::vector<double>> matrix1 = {{1, 2}, {3, 4}};
    std::vector<std::vector<double>> matrix2 = {{5, 6}, {7, 8}};
    
    std::cout << "Matrix Operations:" << std::endl;
    std::cout << "Matrix 1:" << std::endl;
    for (const auto& row : matrix1) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Matrix 2:" << std::endl;
    for (const auto& row : matrix2) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    auto sum = matrix_add(matrix1, matrix2);
    std::cout << "Sum of matrices:" << std::endl;
    for (const auto& row : sum) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    auto product = matrix_multiply(matrix1, matrix2);
    std::cout << "Product of matrices:" << std::endl;
    for (const auto& row : product) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Determinant of Matrix 1: " << matrix_determinant(matrix1) << std::endl;
    
    return 0;
}