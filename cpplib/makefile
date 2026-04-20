CXX      := g++
CXXFLAGS := -O3 -Wall -Wextra -shared -std=c++17 -fPIC $(shell python3 -m pybind11 --includes)

EXT      := $(shell python3-config --extension-suffix)
TARGET   := ../quarto_cpp.so

SRCS     := quarto_env.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f ../quarto_cpp.*