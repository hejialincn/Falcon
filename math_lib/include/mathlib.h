#ifndef MATHLIB_H
#define MATHLIB_H

#include <cmath>
#include <vector>
#include <stdexcept>

namespace MathLib {

// Basic arithmetic operations
double add(double a, double b);
double subtract(double a, double b);
double multiply(double a, double b);
double divide(double a, double b);

// Power and root functions
double power(double base, double exponent);
double square_root(double x);
double cube_root(double x);

// Trigonometric functions
double sine(double angle);
double cosine(double angle);
double tangent(double angle);
double arcsin(double x);
double arccos(double x);
double arctan(double x);
double arctan2(double y, double x);

// Logarithmic functions
double log_natural(double x);
double log_base10(double x);
double log_base(double x, double base);

// Exponential functions
double exponential(double x);

// Constants
const double PI = 3.14159265358979323846;
const double E = 2.71828182845904523536;

// Absolute value and rounding
double absolute(double x);
double round_value(double x);
double floor_value(double x);
double ceil_value(double x);

// Factorial and combinatorics
unsigned long long factorial(unsigned int n);
unsigned long long combination(unsigned int n, unsigned int r);
unsigned long long permutation(unsigned int n, unsigned int r);

// Matrix operations
std::vector<std::vector<double>> matrix_multiply(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b);
    
std::vector<std::vector<double>> matrix_add(
    const std::vector<std::vector<double>>& a,
    const std::vector<std::vector<double>>& b);

double matrix_determinant(const std::vector<std::vector<double>>& matrix);
std::vector<std::vector<double>> matrix_inverse(const std::vector<std::vector<double>>& matrix);

// Statistics functions
double mean(const std::vector<double>& data);
double median(std::vector<double> data);
double standard_deviation(const std::vector<double>& data);
double variance(const std::vector<double>& data);
double max_value(const std::vector<double>& data);
double min_value(const std::vector<double>& data);

} // namespace MathLib

#endif // MATHLIB_H