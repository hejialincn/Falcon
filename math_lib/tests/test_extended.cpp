#include "../include/mathlib.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <complex>

void test_complex_numbers() {
    std::cout << "Testing complex number operations..." << std::endl;
    
    std::complex<double> a(3.0, 4.0);
    std::complex<double> b(1.0, 2.0);
    
    auto result_add = MathLib::complex_add(a, b);
    assert(std::abs(result_add.real() - 4.0) < 1e-9);
    assert(std::abs(result_add.imag() - 6.0) < 1e-9);
    
    auto result_mult = MathLib::complex_multiply(a, b);
    assert(std::abs(result_mult.real() - (-5.0)) < 1e-9);
    assert(std::abs(result_mult.imag() - 10.0) < 1e-9);
    
    std::cout << "Complex number tests passed!" << std::endl;
}

void test_hyperbolic_functions() {
    std::cout << "Testing hyperbolic functions..." << std::endl;
    
    double x = 1.0;
    assert(std::abs(MathLib::sinh(x) - std::sinh(x)) < 1e-9);
    assert(std::abs(MathLib::cosh(x) - std::cosh(x)) < 1e-9);
    assert(std::abs(MathLib::tanh(x) - std::tanh(x)) < 1e-9);
    
    std::cout << "Hyperbolic function tests passed!" << std::endl;
}

void test_special_functions() {
    std::cout << "Testing special functions..." << std::endl;
    
    // Test gamma function
    assert(std::abs(MathLib::gamma_function(4.0) - 6.0) < 1e-6); // Gamma(4) = 3! = 6
    
    // Test beta function
    assert(std::abs(MathLib::beta_function(2.0, 3.0) - 0.0833333) < 1e-5); // Beta(2,3) = 1/12
    
    // Test error function
    assert(std::abs(MathLib::erf_value(0.0)) < 1e-9); // erf(0) = 0
    
    std::cout << "Special function tests passed!" << std::endl;
}

void test_numerical_methods() {
    std::cout << "Testing numerical methods..." << std::endl;
    
    // Test numerical integration: integral of x^2 from 0 to 2 = 8/3
    auto f = [](double x) -> double { return x * x; };
    double integral = MathLib::numerical_integration(f, 0.0, 2.0, 1000);
    assert(std::abs(integral - 8.0/3.0) < 1e-3);
    
    // Test numerical derivative: derivative of x^2 at x=2 is 4
    double deriv = MathLib::numerical_derivative(f, 2.0);
    assert(std::abs(deriv - 4.0) < 1e-3);
    
    std::cout << "Numerical method tests passed!" << std::endl;
}

void test_physics_functions() {
    std::cout << "Testing physics functions..." << std::endl;
    
    assert(std::abs(MathLib::velocity(100.0, 10.0) - 10.0) < 1e-9);
    assert(std::abs(MathLib::force(10.0, 9.8) - 98.0) < 1e-9);
    assert(std::abs(MathLib::kinetic_energy(10.0, 5.0) - 125.0) < 1e-9);
    
    std::cout << "Physics function tests passed!" << std::endl;
}

void test_geometry_functions() {
    std::cout << "Testing geometry functions..." << std::endl;
    
    assert(std::abs(MathLib::area_circle(1.0) - MathLib::PI) < 1e-9);
    assert(std::abs(MathLib::distance_2d(0, 0, 3, 4) - 5.0) < 1e-9);
    
    std::cout << "Geometry function tests passed!" << std::endl;
}

void test_signal_processing() {
    std::cout << "Testing signal processing functions..." << std::endl;
    
    std::vector<double> signal = {1.0, 2.0, 3.0, 4.0};
    auto dft_result = MathLib::dft(signal);
    // Just ensure we get back the right size
    assert(dft_result.size() == signal.size());
    
    std::vector<double> sig1 = {1.0, 2.0};
    std::vector<double> sig2 = {1.0, 1.0};
    auto conv_result = MathLib::convolution(sig1, sig2);
    assert(conv_result.size() == 3);
    assert(std::abs(conv_result[0] - 1.0) < 1e-9);
    assert(std::abs(conv_result[1] - 3.0) < 1e-9);
    
    std::cout << "Signal processing tests passed!" << std::endl;
}

void test_optimization() {
    std::cout << "Testing optimization functions..." << std::endl;
    
    // Minimize f(x) = (x-3)^2, minimum at x=3
    auto f = [](double x) -> double { return (x - 3.0) * (x - 3.0); };
    double min_x = MathLib::golden_section_search(f, 0.0, 10.0, 1e-4);
    assert(std::abs(min_x - 3.0) < 1e-3);
    
    std::cout << "Optimization tests passed!" << std::endl;
}

void test_differential_equations() {
    std::cout << "Testing differential equation solvers..." << std::endl;
    
    // Solve dy/dx = y, y(0) = 1, solution is y = e^x
    auto dydx = [](double x, double y) -> double { return y; };
    auto result = MathLib::euler_method(dydx, 0.0, 1.0, 1.0, 100);
    double approx_e = result.back().second;
    assert(std::abs(approx_e - std::exp(1.0)) < 1e-1); // Less strict tolerance for Euler
    
    std::cout << "Differential equation tests passed!" << std::endl;
}

int main() {
    std::cout << "Running extended MathLib tests..." << std::endl;
    
    test_complex_numbers();
    test_hyperbolic_functions();
    test_special_functions();
    test_numerical_methods();
    test_physics_functions();
    test_geometry_functions();
    test_signal_processing();
    test_optimization();
    test_differential_equations();
    
    std::cout << "All extended tests passed!" << std::endl;
    return 0;
}
