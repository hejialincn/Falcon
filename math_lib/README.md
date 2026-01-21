# MathLib - Comprehensive C++ Math Library

A comprehensive mathematical library written in C++ providing various mathematical functions including arithmetic operations, trigonometric functions, logarithms, factorials, matrix operations, and statistical functions.

## Features

### Basic Arithmetic Operations
- Addition, subtraction, multiplication, division
- Proper error handling for division by zero

### Power and Root Functions
- Power calculations (base^exponent)
- Square root and cube root
- Error handling for invalid inputs

### Trigonometric Functions
- Sine, cosine, tangent
- Arcsine, arccosine, arctangent
- atan2 function for coordinate conversion

### Logarithmic Functions
- Natural logarithm (base e)
- Base-10 logarithm
- Arbitrary base logarithm
- Error handling for invalid inputs

### Exponential Functions
- Exponential function (e^x)

### Absolute Value and Rounding
- Absolute value
- Round, floor, ceiling functions

### Combinatorics
- Factorial calculation
- Combination (nCr) and permutation (nPr) functions

### Matrix Operations
- Matrix addition and multiplication
- Matrix determinant calculation
- Matrix inversion
- Dimension validation

### Statistical Functions
- Mean, median calculations
- Standard deviation and variance
- Maximum and minimum values
- Error handling for empty datasets

## Usage

### Building the Library

```bash
make all                    # Build the library and example
make test                   # Build the test suite
make run_tests              # Run all tests
make run_example            # Run the example program
make clean                  # Remove build artifacts
```

### Using the Library

Include the header in your project:
```cpp
#include "include/mathlib.h"

using namespace MathLib;

int main() {
    double result = add(5.0, 3.0);
    double sqrt_val = square_root(16.0);
    // ... use other functions
    
    return 0;
}
```

## Examples

See `example.cpp` for usage examples of all major functionality.

## Testing

The library includes comprehensive unit tests covering all major functions. Run tests with:
```bash
make run_tests
```

## Directory Structure

```
math_lib/
├── include/           # Header files
│   └── mathlib.h      # Main library header
├── src/               # Source implementations
│   └── mathlib.cpp    # Implementation file
├── tests/             # Test files
│   └── test_mathlib.cpp
├── example.cpp        # Usage example
├── Makefile           # Build configuration
└── README.md          # This file
```

## Requirements

- C++11 or later
- GNU Make
- G++ compiler

## License

This library is open-source and available under the MIT license.