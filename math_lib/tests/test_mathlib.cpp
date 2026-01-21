#include "../include/mathlib.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace MathLib;

bool almost_equal(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

void test_basic_arithmetic() {
    std::cout << "Testing basic arithmetic operations..." << std::endl;
    
    assert(add(2.0, 3.0) == 5.0);
    assert(subtract(5.0, 3.0) == 2.0);
    assert(multiply(4.0, 3.0) == 12.0);
    assert(divide(10.0, 2.0) == 5.0);
    
    try {
        divide(5.0, 0.0);
        assert(false); // Should not reach here
    } catch (const std::domain_error&) {
        // Expected
    }
    
    std::cout << "Basic arithmetic tests passed!" << std::endl;
}

void test_power_and_roots() {
    std::cout << "Testing power and root functions..." << std::endl;
    
    assert(power(2.0, 3.0) == 8.0);
    assert(almost_equal(square_root(16.0), 4.0));
    assert(almost_equal(cube_root(27.0), 3.0));
    
    try {
        square_root(-1.0);
        assert(false); // Should not reach here
    } catch (const std::domain_error&) {
        // Expected
    }
    
    std::cout << "Power and root tests passed!" << std::endl;
}

void test_trigonometry() {
    std::cout << "Testing trigonometric functions..." << std::endl;
    
    assert(almost_equal(sine(0.0), 0.0));
    assert(almost_equal(cosine(0.0), 1.0));
    assert(almost_equal(tangent(0.0), 0.0));
    
    // Test with PI/2 (approximately 1.5708)
    double pi_half = PI / 2.0;
    assert(almost_equal(sine(pi_half), 1.0, 1e-6));
    assert(almost_equal(cosine(pi_half), 0.0, 1e-6));
    
    // Test inverse functions
    assert(almost_equal(arcsin(0.0), 0.0));
    assert(almost_equal(arccos(1.0), 0.0));
    assert(almost_equal(arctan(0.0), 0.0));
    
    std::cout << "Trigonometric tests passed!" << std::endl;
}

void test_logarithms() {
    std::cout << "Testing logarithmic functions..." << std::endl;
    
    assert(almost_equal(log_natural(E), 1.0));
    assert(almost_equal(log_base10(100.0), 2.0));
    assert(almost_equal(log_base(8.0, 2.0), 3.0));
    
    try {
        log_natural(-1.0);
        assert(false); // Should not reach here
    } catch (const std::domain_error&) {
        // Expected
    }
    
    std::cout << "Logarithmic tests passed!" << std::endl;
}

void test_exponential() {
    std::cout << "Testing exponential function..." << std::endl;
    
    assert(almost_equal(exponential(0.0), 1.0));
    assert(almost_equal(exponential(1.0), E));
    
    std::cout << "Exponential tests passed!" << std::endl;
}

void test_abs_rounding() {
    std::cout << "Testing absolute value and rounding functions..." << std::endl;
    
    assert(absolute(-5.0) == 5.0);
    assert(absolute(5.0) == 5.0);
    
    assert(round_value(3.7) == 4.0);
    assert(round_value(3.2) == 3.0);
    
    assert(floor_value(3.7) == 3.0);
    assert(ceil_value(3.2) == 4.0);
    
    std::cout << "Absolute value and rounding tests passed!" << std::endl;
}

void test_factorial_combinatorics() {
    std::cout << "Testing factorial and combinatorics functions..." << std::endl;
    
    assert(factorial(0) == 1);
    assert(factorial(1) == 1);
    assert(factorial(5) == 120);
    
    assert(combination(5, 2) == 10);
    assert(combination(10, 0) == 1);
    assert(combination(10, 10) == 1);
    
    assert(permutation(5, 2) == 20);
    assert(permutation(10, 0) == 1);
    
    std::cout << "Factorial and combinatorics tests passed!" << std::endl;
}

void test_matrix_operations() {
    std::cout << "Testing matrix operations..." << std::endl;
    
    // Test matrix addition
    std::vector<std::vector<double>> mat1 = {{1, 2}, {3, 4}};
    std::vector<std::vector<double>> mat2 = {{5, 6}, {7, 8}};
    std::vector<std::vector<double>> expected_add = {{6, 8}, {10, 12}};
    auto result_add = matrix_add(mat1, mat2);
    
    for(size_t i = 0; i < expected_add.size(); ++i) {
        for(size_t j = 0; j < expected_add[i].size(); ++j) {
            assert(result_add[i][j] == expected_add[i][j]);
        }
    }
    
    // Test matrix multiplication
    std::vector<std::vector<double>> mat3 = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<double>> mat4 = {{7, 8}, {9, 10}, {11, 12}};
    std::vector<std::vector<double>> expected_mult = {{58, 64}, {139, 154}};
    auto result_mult = matrix_multiply(mat3, mat4);
    
    for(size_t i = 0; i < expected_mult.size(); ++i) {
        for(size_t j = 0; j < expected_mult[i].size(); ++j) {
            assert(almost_equal(result_mult[i][j], expected_mult[i][j]));
        }
    }
    
    // Test determinant
    std::vector<std::vector<double>> mat_det = {{4, 3}, {6, 3}};
    double expected_det = -6.0;
    assert(almost_equal(matrix_determinant(mat_det), expected_det));
    
    std::cout << "Matrix operation tests passed!" << std::endl;
}

void test_statistics() {
    std::cout << "Testing statistics functions..." << std::endl;
    
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    assert(mean(data) == 3.0);
    assert(median(data) == 3.0);
    assert(almost_equal(variance(data), 2.0));
    assert(max_value(data) == 5.0);
    assert(min_value(data) == 1.0);
    
    std::cout << "Statistics tests passed!" << std::endl;
}

int main() {
    std::cout << "Running Math Library Tests..." << std::endl;
    
    test_basic_arithmetic();
    test_power_and_roots();
    test_trigonometry();
    test_logarithms();
    test_exponential();
    test_abs_rounding();
    test_factorial_combinatorics();
    test_matrix_operations();
    test_statistics();
    
    std::cout << "All tests passed successfully!" << std::endl;
    
    return 0;
}