SRCS = $(wildcard *.cu)
OBJS = $(patsubst %.cu, $(BUILD_DIR)/%.o, $(SRCS))

all: $(BUILD_DIR) $(OBJS)

$(BUILD_DIR)/%.o: %.cu
	@$(NVXX) $^ -c -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
