SRCS = $(wildcard *.c)
OBJS = $(patsubst %.c, $(BUILD_DIR)/%.o, $(SRCS))

all: $(BUILD_DIR) $(OBJS)

$(BUILD_DIR)/%.o: %.c
	@$(GXX) $^ $(CFLAGS) -c -o $@

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
