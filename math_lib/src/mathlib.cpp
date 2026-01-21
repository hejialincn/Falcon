#include "../include/mathlib.h"
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <random>

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
// Complex number operations
std::complex<double> complex_add(const std::complex<double>& a, const std::complex<double>& b) {
    return a + b;
}

std::complex<double> complex_subtract(const std::complex<double>& a, const std::complex<double>& b) {
    return a - b;
}

std::complex<double> complex_multiply(const std::complex<double>& a, const std::complex<double>& b) {
    return a * b;
}

std::complex<double> complex_divide(const std::complex<double>& a, const std::complex<double>& b) {
    if (std::abs(b) < 1e-10) {
        throw std::domain_error("Division by zero in complex division");
    }
    return a / b;
}

std::complex<double> complex_power(const std::complex<double>& base, const std::complex<double>& exponent) {
    return std::pow(base, exponent);
}

std::complex<double> complex_sqrt(const std::complex<double>& z) {
    return std::sqrt(z);
}

double complex_abs(const std::complex<double>& z) {
    return std::abs(z);
}

double complex_arg(const std::complex<double>& z) {
    return std::arg(z);
}

// Advanced trigonometric and hyperbolic functions
double sinh(double x) {
    return std::sinh(x);
}

double cosh(double x) {
    return std::cosh(x);
}

double tanh(double x) {
    return std::tanh(x);
}

double asinh(double x) {
    return std::asinh(x);
}

double acosh(double x) {
    if (x < 1.0) {
        throw std::domain_error("acosh domain error: x must be >= 1");
    }
    return std::acosh(x);
}

double atanh(double x) {
    if (x <= -1.0 || x >= 1.0) {
        throw std::domain_error("atanh domain error: |x| must be < 1");
    }
    return std::atanh(x);
}

// Special functions
double gamma_function(double x) {
    if (x <= 0 && std::floor(x) == x) {
        throw std::domain_error("Gamma function undefined for non-positive integers");
    }
    return std::tgamma(x);
}

double beta_function(double x, double y) {
    return std::exp(std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y));
}

double erf_value(double x) {
    return std::erf(x);
}

double erfc_value(double x) {
    return std::erfc(x);
}

// Numerical integration and differentiation
double numerical_integration(std::function<double(double)> f, double a, double b, int n) {
    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    
    double h = (b - a) / n;
    double sum = 0.0;
    
    for (int i = 0; i < n; ++i) {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;
        sum += (f(x1) + f(x2)) * h / 2.0;
    }
    
    return sum;
}

double numerical_derivative(std::function<double(double)> f, double x, double h) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

// Interpolation methods
double linear_interpolation(double x0, double y0, double x1, double y1, double x) {
    if (std::abs(x1 - x0) < 1e-10) {
        throw std::domain_error("Linear interpolation: x1 and x0 cannot be equal");
    }
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

std::vector<double> polynomial_interpolation(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& xi) {
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("x and y vectors must be of equal size and non-empty");
    }
    
    std::vector<double> result;
    for (double x_val : xi) {
        double y_val = 0.0;
        
        for (size_t i = 0; i < x.size(); ++i) {
            double term = y[i];
            for (size_t j = 0; j < x.size(); ++j) {
                if (i != j) {
                    term *= (x_val - x[j]) / (x[i] - x[j]);
                }
            }
            y_val += term;
        }
        result.push_back(y_val);
    }
    
    return result;
}

// Physics and engineering functions
double velocity(double displacement, double time) {
    if (std::abs(time) < 1e-10) {
        throw std::domain_error("Time cannot be zero for velocity calculation");
    }
    return displacement / time;
}

double acceleration(double initial_velocity, double final_velocity, double time) {
    if (std::abs(time) < 1e-10) {
        throw std::domain_error("Time cannot be zero for acceleration calculation");
    }
    return (final_velocity - initial_velocity) / time;
}

double force(double mass, double acceleration) {
    return mass * acceleration;
}

double kinetic_energy(double mass, double velocity) {
    return 0.5 * mass * velocity * velocity;
}

double potential_energy(double mass, double gravity, double height) {
    return mass * gravity * height;
}

double work_done(double force, double distance, double angle_radians) {
    return force * distance * std::cos(angle_radians);
}

double momentum(double mass, double velocity) {
    return mass * velocity;
}

double angular_velocity(double theta, double time) {
    if (std::abs(time) < 1e-10) {
        throw std::domain_error("Time cannot be zero for angular velocity calculation");
    }
    return theta / time;
}

double centripetal_acceleration(double velocity, double radius) {
    if (std::abs(radius) < 1e-10) {
        throw std::domain_error("Radius cannot be zero for centripetal acceleration calculation");
    }
    return (velocity * velocity) / radius;
}

double frequency_to_angular_frequency(double freq) {
    return 2.0 * MathLib::PI * freq;
}

double wavelength(double velocity, double frequency) {
    if (std::abs(frequency) < 1e-10) {
        throw std::domain_error("Frequency cannot be zero for wavelength calculation");
    }
    return velocity / frequency;
}

// Geometry functions
double area_circle(double radius) {
    if (radius < 0) {
        throw std::domain_error("Radius cannot be negative");
    }
    return MathLib::PI * radius * radius;
}

double circumference_circle(double radius) {
    if (radius < 0) {
        throw std::domain_error("Radius cannot be negative");
    }
    return 2.0 * MathLib::PI * radius;
}

double area_rectangle(double length, double width) {
    if (length < 0 || width < 0) {
        throw std::domain_error("Length and width cannot be negative");
    }
    return length * width;
}

double area_triangle(double base, double height) {
    if (base < 0 || height < 0) {
        throw std::domain_error("Base and height cannot be negative");
    }
    return 0.5 * base * height;
}

double volume_sphere(double radius) {
    if (radius < 0) {
        throw std::domain_error("Radius cannot be negative");
    }
    return (4.0/3.0) * MathLib::PI * radius * radius * radius;
}

double surface_area_sphere(double radius) {
    if (radius < 0) {
        throw std::domain_error("Radius cannot be negative");
    }
    return 4.0 * MathLib::PI * radius * radius;
}

double distance_2d(double x1, double y1, double x2, double y2) {
    return std::sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

double distance_3d(double x1, double y1, double z1, double x2, double y2, double z2) {
    return std::sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1) + (z2 - z1)*(z2 - z1));
}

// Signal processing functions
std::vector<double> dft(const std::vector<double>& signal) {
    int N = signal.size();
    std::vector<double> result(N);
    
    for (int k = 0; k < N; ++k) {
        double real_part = 0.0;
        for (int n = 0; n < N; ++n) {
            double angle = 2.0 * MathLib::PI * k * n / N;
            real_part += signal[n] * std::cos(angle);
        }
        result[k] = real_part;
    }
    
    return result;
}

std::vector<double> convolution(const std::vector<double>& signal1, const std::vector<double>& signal2) {
    int size1 = signal1.size();
    int size2 = signal2.size();
    int result_size = size1 + size2 - 1;
    std::vector<double> result(result_size, 0.0);
    
    for (int i = 0; i < result_size; ++i) {
        for (int j = 0; j < size1; ++j) {
            if ((i - j) >= 0 && (i - j) < size2) {
                result[i] += signal1[j] * signal2[i - j];
            }
        }
    }
    
    return result;
}

// Optimization algorithms
double golden_section_search(std::function<double(double)> f, double a, double b, double tolerance) {
    const double golden_ratio = (std::sqrt(5.0) - 1.0) / 2.0;  // ~0.618
    
    double c = b - golden_ratio * (b - a);
    double d = a + golden_ratio * (b - a);
    
    while (std::abs(c - d) > tolerance) {
        if (f(c) < f(d)) {
            b = d;
        } else {
            a = c;
        }
        
        c = b - golden_ratio * (b - a);
        d = a + golden_ratio * (b - a);
    }
    
    return (b + a) / 2.0;
}

// Differential equation solvers
std::vector<std::pair<double, double>> euler_method(std::function<double(double, double)> dy_dx, double x0, double y0, double xn, int steps) {
    if (steps <= 0) {
        throw std::invalid_argument("Steps must be positive");
    }
    
    std::vector<std::pair<double, double>> result;
    double h = (xn - x0) / steps;
    double x = x0;
    double y = y0;
    
    result.push_back({x, y});
    
    for (int i = 0; i < steps; ++i) {
        y = y + h * dy_dx(x, y);
        x = x + h;
        result.push_back({x, y});
    }
    
    return result;
}

std::vector<std::pair<double, double>> rk4_method(std::function<double(double, double)> dy_dx, double x0, double y0, double xn, int steps) {
    if (steps <= 0) {
        throw std::invalid_argument("Steps must be positive");
    }
    
    std::vector<std::pair<double, double>> result;
    double h = (xn - x0) / steps;
    double x = x0;
    double y = y0;
    
    result.push_back({x, y});
    
    for (int i = 0; i < steps; ++i) {
        double k1 = h * dy_dx(x, y);
        double k2 = h * dy_dx(x + h/2.0, y + k1/2.0);
        double k3 = h * dy_dx(x + h/2.0, y + k2/2.0);
        double k4 = h * dy_dx(x + h, y + k3);
        
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0;
        x = x + h;
        result.push_back({x, y});
    }
    
    return result;
}

// Performance optimization utilities
template<typename T>
T fast_inverse_sqrt(T x) {
    if (x <= 0) {
        throw std::domain_error("Fast inverse sqrt requires positive input");
    }
    
    // Quake III Arena fast inverse square root algorithm
    float y = static_cast<float>(x);
    float x2 = y * 0.5F;
    int32_t i = *(int32_t*)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;
    y = y * (1.5F - (x2 * y * y));   // Newton-Raphson iteration
    return static_cast<T>(y);
}

} // namespace MathLib
