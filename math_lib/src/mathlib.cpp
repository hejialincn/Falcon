#include "../include/mathlib.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace MathLib {

// Basic arithmetic operations
double add(double a, double b) {
    return a + b;
}

double subtract(double a, double b) {
    return a - b;
}

double multiply(double a, double b) {
    return a * b;
}

double divide(double a, double b) {
    if (b == 0) {
        throw std::domain_error("Division by zero");
    }
    return a / b;
}

// Power and root functions
double power(double base, double exponent) {
    return std::pow(base, exponent);
}

double square_root(double x) {
    if (x < 0) {
        throw std::domain_error("Square root of negative number");
    }
    return std::sqrt(x);
}

double cube_root(double x) {
    return std::cbrt(x);
}

// Trigonometric functions
double sine(double angle) {
    return std::sin(angle);
}

double cosine(double angle) {
    return std::cos(angle);
}

double tangent(double angle) {
    return std::tan(angle);
}

double arcsin(double x) {
    if (x < -1 || x > 1) {
        throw std::domain_error("Arcsin input must be between -1 and 1");
    }
    return std::asin(x);
}

double arccos(double x) {
    if (x < -1 || x > 1) {
        throw std::domain_error("Arccos input must be between -1 and 1");
    }
    return std::acos(x);
}

double arctan(double x) {
    return std::atan(x);
}

double arctan2(double y, double x) {
    return std::atan2(y, x);
}

// Logarithmic functions
double log_natural(double x) {
    if (x <= 0) {
        throw std::domain_error("Logarithm input must be positive");
    }
    return std::log(x);
}

double log_base10(double x) {
    if (x <= 0) {
        throw std::domain_error("Logarithm input must be positive");
    }
    return std::log10(x);
}

double log_base(double x, double base) {
    if (x <= 0 || base <= 0 || base == 1) {
        throw std::domain_error("Invalid inputs for logarithm base calculation");
    }
    return std::log(x) / std::log(base);
}

// Exponential functions
double exponential(double x) {
    return std::exp(x);
}

// Absolute value and rounding
double absolute(double x) {
    return std::abs(x);
}

double round_value(double x) {
    return std::round(x);
}

double floor_value(double x) {
    return std::floor(x);
}

double ceil_value(double x) {
    return std::ceil(x);
}

// Factorial and combinatorics
unsigned long long factorial(unsigned int n) {
    if (n == 0 || n == 1) {
        return 1;
    }
    
    unsigned long long result = 1;
    for (unsigned int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

unsigned long long combination(unsigned int n, unsigned int r) {
    if (r > n) {
        return 0;
    }
    if (r == 0 || r == n) {
        return 1;
    }
    
    // Optimize to avoid overflow
    unsigned long long result = 1;
    for (unsigned int i = 0; i < r; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

unsigned long long permutation(unsigned int n, unsigned int r) {
    if (r > n) {
        return 0;
    }
    return factorial(n) / factorial(n - r);
}

// Matrix operations helper function to get matrix dimensions
size_t matrix_rows(const std::vector<std::vector<double>>& matrix) {
    return matrix.size();
}

size_t matrix_cols(const std::vector<std::vector<double>>& matrix) {
    return matrix.empty() ? 0 : matrix[0].size();
}

std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b) {
    
    size_t rows_a = matrix_rows(a);
    size_t cols_a = matrix_cols(a);
    size_t rows_b = matrix_rows(b);
    size_t cols_b = matrix_cols(b);
    
    if (cols_a != rows_b) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    std::vector<std::vector<double>> result(rows_a, std::vector<double>(cols_b, 0.0));
    
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_b; ++j) {
            for (size_t k = 0; k < cols_a; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    
    return result;
}

std::vector<std::vector<double>> matrix_add(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b) {
    
    size_t rows_a = matrix_rows(a);
    size_t cols_a = matrix_cols(a);
    size_t rows_b = matrix_rows(b);
    size_t cols_b = matrix_cols(b);
    
    if (rows_a != rows_b || cols_a != cols_b) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    std::vector<std::vector<double>> result(rows_a, std::vector<double>(cols_a));
    
    for (size_t i = 0; i < rows_a; ++i) {
        for (size_t j = 0; j < cols_a; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    
    return result;
}

// Helper function to create submatrix for determinant calculation
std::vector<std::vector<double>> get_submatrix(
    const std::vector<std::vector<double>>& matrix,
    size_t exclude_row,
    size_t exclude_col) {
    
    std::vector<std::vector<double>> submatrix;
    for (size_t i = 0; i < matrix.size(); ++i) {
        if (i == exclude_row) continue;
        
        std::vector<double> row;
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            if (j == exclude_col) continue;
            row.push_back(matrix[i][j]);
        }
        submatrix.push_back(row);
    }
    return submatrix;
}

double matrix_determinant(const std::vector<std::vector<double>>& matrix) {
    size_t n = matrix.size();
    
    if (n != matrix_cols(matrix)) {
        throw std::invalid_argument("Matrix must be square for determinant calculation");
    }
    
    // Base cases
    if (n == 1) {
        return matrix[0][0];
    } else if (n == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    
    // Recursive case using cofactor expansion along the first row
    double det = 0;
    for (size_t j = 0; j < n; ++j) {
        auto submatrix = get_submatrix(matrix, 0, j);
        double sign = (j % 2 == 0) ? 1 : -1;
        det += sign * matrix[0][j] * matrix_determinant(submatrix);
    }
    
    return det;
}

std::vector<std::vector<double>> matrix_inverse(const std::vector<std::vector<double>>& matrix) {
    size_t n = matrix.size();
    
    if (n != matrix_cols(matrix)) {
        throw std::invalid_argument("Matrix must be square for inverse calculation");
    }
    
    double det = matrix_determinant(matrix);
    if (std::abs(det) < 1e-10) {
        throw std::runtime_error("Matrix is singular, inverse does not exist");
    }
    
    // For 1x1 matrix
    if (n == 1) {
        return {{1.0 / matrix[0][0]}};
    }
    
    // Create the adjugate matrix
    std::vector<std::vector<double>> adjugate(n, std::vector<double>(n));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            auto submatrix = get_submatrix(matrix, i, j);
            double sign = ((i + j) % 2 == 0) ? 1 : -1;
            adjugate[j][i] = sign * matrix_determinant(submatrix); // Transpose is built-in
        }
    }
    
    // Divide by determinant
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            adjugate[i][j] /= det;
        }
    }
    
    return adjugate;
}

// Statistics functions
double mean(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::domain_error("Cannot calculate mean of empty dataset");
    }
    
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double median(std::vector<double> data) {
    if (data.empty()) {
        throw std::domain_error("Cannot calculate median of empty dataset");
    }
    
    std::sort(data.begin(), data.end());
    size_t size = data.size();
    
    if (size % 2 == 0) {
        return (data[size/2 - 1] + data[size/2]) / 2.0;
    } else {
        return data[size/2];
    }
}

double standard_deviation(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::domain_error("Cannot calculate standard deviation of empty dataset");
    }
    
    double avg = mean(data);
    double sum_squared_diff = 0.0;
    
    for (double val : data) {
        double diff = val - avg;
        sum_squared_diff += diff * diff;
    }
    
    return std::sqrt(sum_squared_diff / data.size());
}

double variance(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::domain_error("Cannot calculate variance of empty dataset");
    }
    
    double avg = mean(data);
    double sum_squared_diff = 0.0;
    
    for (double val : data) {
        double diff = val - avg;
        sum_squared_diff += diff * diff;
    }
    
    return sum_squared_diff / data.size();
}

double max_value(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::domain_error("Cannot find max of empty dataset");
    }
    
    return *std::max_element(data.begin(), data.end());
}

double min_value(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::domain_error("Cannot find min of empty dataset");
    }
    
    return *std::min_element(data.begin(), data.end());
}

} // namespace MathLib