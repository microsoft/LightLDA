PROJECT := $(shell readlink $(dir $(lastword $(MAKEFILE_LIST))) -f)

CXX = g++
CXXFLAGS = -O3 \
           -std=c++11 \
           -Wall \
           -Wno-sign-compare \
           -fno-omit-frame-pointer

MULTIVERSO_DIR = $(PROJECT)/multiverso
MULTIVERSO_INC = $(MULTIVERSO_DIR)/include
MULTIVERSO_LIB = $(MULTIVERSO_DIR)/lib
THIRD_PARTY_LIB = $(MULTIVERSO_DIR)/third_party/lib

INC_FLAGS = -I$(MULTIVERSO_INC)
LD_FLAGS  = -L$(MULTIVERSO_LIB) -lmultiverso
LD_FLAGS += -L$(THIRD_PARTY_LIB) -lzmq -lmpich -lmpl -lpthread
  	  	
LIGHTLDA_HEADERS = $(shell find $(PROJECT)/src -type f -name "*.h")
LIGHTLDA_SRC     = $(shell find $(PROJECT)/src -type f -name "*.cpp")
LIGHTLDA_OBJ = $(LIGHTLDA_SRC:.cpp=.o)

DUMP_BINARY_SRC = $(shell find $(PROJECT)/preprocess -type f -name "*.cpp")

BIN_DIR = $(PROJECT)/bin
LIGHTLDA = $(BIN_DIR)/lightlda
DUMP_BINARY = $(BIN_DIR)/dump_binary

all: path \
	 lightlda \
	 dump_binary

path: $(BIN_DIR)

$(BIN_DIR):
	mkdir -p $@

$(LIGHTLDA): $(LIGHTLDA_OBJ)
	$(CXX) $(LIGHTLDA_OBJ) $(CXXFLAGS) $(INC_FLAGS) $(LD_FLAGS) -o $@

$(LIGHTLDA_OBJ): %.o: %.cpp $(LIGHTLDA_HEADERS) $(MULTIVERSO_INC)
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

$(DUMP_BINARY): $(DUMP_BINARY_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

lightlda: path $(LIGHTLDA)
	
dump_binary: path $(DUMP_BINARY)

clean:
	rm -rf $(BIN_DIR) $(LIGHTLDA_OBJ)

.PHONY: all path lightlda dump_binary clean
