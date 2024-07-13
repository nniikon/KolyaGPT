include config.mk
 
.PHONY: all source mnist mnist_parser ftb main build clean

all: source mnist mnist_parser ftb main
ifeq ($(GPU),1)
	@$(MAKE) -C ./chubarov_lib/chubarov_cuda/
	@$(NVXX) -o $(BUILD_DIR)/$(EXEC_NAME) $(BUILD_DIR)/*.o $(NFLAGS)
else
	@$(MAKE) -C ./chubarov_lib/chubarov_cpu/
	@$(GXX) -o $(BUILD_DIR)/$(EXEC_NAME) $(BUILD_DIR)/*.o
endif

source:
	@$(MAKE) -C ./source/ $(MAKECMDGOALS)

mnist:
	@$(MAKE) -C ./mnist/ $(MAKECMDGOALS)

mnist_parser:
	@$(MAKE) -C ./mnist/mnist_parser/ $(MAKECMDGOALS)

ftb:
	@$(MAKE) -C ./mnist/mnist_parser/file_to_buffer/ $(MAKECMDGOALS)

main:
	@$(GXX) main.cpp $(CFLAGS) -c -o $(BUILD_DIR)/main.o


run:
	$(BUILD_DIR)/$(EXEC_NAME)

clean:
	@rm -rf $(BUILD_DIR)
