CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -g
INCLUDES = -I./include
SRCDIR = src
TESTDIR = tests
SOURCES = $(SRCDIR)/mathlib.cpp
TEST_SOURCES = $(TESTDIR)/test_mathlib.cpp
EXAMPLE_SOURCE = example.cpp
TARGET_LIB = libmathlib.a
TEST_TARGET = test_mathlib
EXAMPLE_TARGET = example

# Default target
all: $(TARGET_LIB) $(EXAMPLE_TARGET)

# Build static library
$(TARGET_LIB): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $^
	ar rcs $(TARGET_LIB) *.o
	rm -f *.o

# Build test executable
test: $(TARGET_LIB) $(TEST_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TEST_TARGET) $(TEST_SOURCES) $(TARGET_LIB)

# Build example executable
$(EXAMPLE_TARGET): $(TARGET_LIB) $(EXAMPLE_SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXAMPLE_TARGET) $(EXAMPLE_SOURCE) $(TARGET_LIB)

# Run tests
run_tests: test
	./$(TEST_TARGET)

# Run example
run_example: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET_LIB) $(TEST_TARGET) $(EXAMPLE_TARGET) *.o

# Rebuild everything
rebuild: clean all

.PHONY: all test run_tests run_example clean rebuild