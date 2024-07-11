include config.mk
 
all:
	@$(MAKE) -C ./source/
	@$(MAKE) -C ./mnist/
	@$(MAKE) -C ./mnist/mnist_parser/
	@$(MAKE) -C ./mnist/mnist_parser/file_to_buffer
	@$(GXX) main.cpp $(CFLAGS) -c -o $(BUILD_DIR)/main.o
	@$(GXX) $(CFLAGS) -o $(BUILD_DIR)/$(EXEC_NAME) $(BUILD_DIR)/*.o

run:
	$(BUILD_DIR)/$(EXEC_NAME)

clean:
	@rm -rf $(BUILD_DIR)
