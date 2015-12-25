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

INC_FLAGS = -I$(MULTIVERSO_INC) -I$(PROJECT)/src -I$(PROJECT)/inference
LD_FLAGS  = -L$(MULTIVERSO_LIB) -lmultiverso
LD_FLAGS += -L$(THIRD_PARTY_LIB) -lzmq -lmpich -lmpl -lpthread
  	  	
BASE_SRC = $(shell find $(PROJECT)/src -type f -name "*.cpp" -type f ! -name "lightlda.cpp")
BASE_OBJ = $(BASE_SRC:.cpp=.o)

LIGHTLDA_HEADERS = $(shell find $(PROJECT)/src -type f -name "*.h")
LIGHTLDA_SRC     = $(shell find $(PROJECT)/src -type f -name "*.cpp")
LIGHTLDA_OBJ = $(LIGHTLDA_SRC:.cpp=.o)

INFER_HEADERS = $(shell find $(PROJECT)/inference -type f -name "*.h")
INFER_SRC = $(shell find $(PROJECT)/inference -type f -name "*.cpp")
INFER_OBJ = $(INFER_SRC:.cpp=.o)

DUMP_BINARY_SRC = $(shell find $(PROJECT)/preprocess -type f -name "*.cpp")

BIN_DIR = $(PROJECT)/bin
LIGHTLDA = $(BIN_DIR)/lightlda
INFER = $(BIN_DIR)/infer
DUMP_BINARY = $(BIN_DIR)/dump_binary

all: path \
	 lightlda \
	 infer \
	 dump_binary

path: $(BIN_DIR)

$(BIN_DIR):
	mkdir -p $@

$(LIGHTLDA): $(LIGHTLDA_OBJ)
	$(CXX) $(LIGHTLDA_OBJ) $(CXXFLAGS) $(INC_FLAGS) $(LD_FLAGS) -o $@

$(LIGHTLDA_OBJ): %.o: %.cpp $(LIGHTLDA_HEADERS) $(MULTIVERSO_INC)
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

$(INFER): $(INFER_OBJ) $(BASE_OBJ)
	$(CXX) $(INFER_OBJ) $(BASE_OBJ) $(CXXFLAGS) $(INC_FLAGS) $(LD_FLAGS) -o $@

$(INFER_OBJ): %.o: %.cpp $(INFER_HEADERS) $(MULTIVERSO_INC)
	$(CXX) $(CXXFLAGS) $(INC_FLAGS) -c $< -o $@

$(DUMP_BINARY): $(DUMP_BINARY_SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

lightlda: path $(LIGHTLDA)

infer: path $(INFER)
	
dump_binary: path $(DUMP_BINARY)

clean:
	rm -rf $(BIN_DIR) $(LIGHTLDA_OBJ) $(INFER_OBJ)

.PHONY: all path lightlda infer dump_binary clean
