#ifndef MATHLIB_H
#define MATHLIB_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <complex>
#include <limits>
#include <functional>

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

// Complex number operations
std::complex<double> complex_add(const std::complex<double>& a, const std::complex<double>& b);
std::complex<double> complex_subtract(const std::complex<double>& a, const std::complex<double>& b);
std::complex<double> complex_multiply(const std::complex<double>& a, const std::complex<double>& b);
std::complex<double> complex_divide(const std::complex<double>& a, const std::complex<double>& b);
std::complex<double> complex_power(const std::complex<double>& base, const std::complex<double>& exponent);
std::complex<double> complex_sqrt(const std::complex<double>& z);
double complex_abs(const std::complex<double>& z);
double complex_arg(const std::complex<double>& z);

// Advanced trigonometric and hyperbolic functions
double sinh(double x);
double cosh(double x);
double tanh(double x);
double asinh(double x);
double acosh(double x);
double atanh(double x);

// Special functions
double gamma_function(double x);
double beta_function(double x, double y);
double erf_value(double x);      // Error function
double erfc_value(double x);     // Complementary error function

// Numerical integration and differentiation
double numerical_integration(std::function<double(double)> f, double a, double b, int n = 1000);
double numerical_derivative(std::function<double(double)> f, double x, double h = 1e-5);

// Interpolation methods
double linear_interpolation(double x0, double y0, double x1, double y1, double x);
std::vector<double> polynomial_interpolation(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& xi);

// Physics and engineering functions
double velocity(double displacement, double time);
double acceleration(double initial_velocity, double final_velocity, double time);
double force(double mass, double acceleration);
double kinetic_energy(double mass, double velocity);
double potential_energy(double mass, double gravity, double height);
double work_done(double force, double distance, double angle_radians = 0);
double momentum(double mass, double velocity);
double angular_velocity(double theta, double time);
double centripetal_acceleration(double velocity, double radius);
double frequency_to_angular_frequency(double freq);
double wavelength(double velocity, double frequency);

// Geometry functions
double area_circle(double radius);
double circumference_circle(double radius);
double area_rectangle(double length, double width);
double area_triangle(double base, double height);
double volume_sphere(double radius);
double surface_area_sphere(double radius);
double distance_2d(double x1, double y1, double x2, double y2);
double distance_3d(double x1, double y1, double z1, double x2, double y2, double z2);

// Signal processing functions
std::vector<double> dft(const std::vector<double>& signal);
std::vector<double> convolution(const std::vector<double>& signal1, const std::vector<double>& signal2);

// Optimization algorithms
double golden_section_search(std::function<double(double)> f, double a, double b, double tolerance = 1e-6);

// Differential equation solvers
std::vector<std::pair<double, double>> euler_method(std::function<double(double, double)> dy_dx, double x0, double y0, double xn, int steps);
std::vector<std::pair<double, double>> rk4_method(std::function<double(double, double)> dy_dx, double x0, double y0, double xn, int steps);

// Performance optimization utilities
template<typename T>
T fast_inverse_sqrt(T x);  // Fast inverse square root (Quake algorithm)

} // namespace MathLib

#endif // MATHLIB_H
