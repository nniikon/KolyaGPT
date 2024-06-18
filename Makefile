include config.mk
 
all:
	@$(GXX) main.cpp $(CFLAGS) -c -o $(BUILD_DIR)/main.o
	@$(MAKE) -C ./source/
	@$(GXX) $(CFLAGS) -o $(BUILD_DIR)/$(EXEC_NAME) $(BUILD_DIR)/*.o

run:
	$(BUILD_DIR)/$(EXEC_NAME)

