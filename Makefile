NVCC=nvcc
CXXFLAGS=-O3 -std=c++14
LIBS=-lcrypto

INCLUDES=-I./include
SRC_DIR=src
BUILD_DIR=build

SOURCES=$(wildcard $(SRC_DIR)/*.cu)
OBJECTS=$(SOURCES:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

TARGET=argon2_miner

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $^ -o $@ $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
