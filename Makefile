include config.mk
 
all:
	@$(MAKE) -C ./source/
	@$(GXX) main.cpp $(CFLAGS) -c -o $(BUILD_DIR)/main.o
	@$(GXX) $(CFLAGS) -o $(BUILD_DIR)/$(EXEC_NAME) $(BUILD_DIR)/*.o

run:
	$(BUILD_DIR)/$(EXEC_NAME)

clean:
	@rm -rf $(BUILD_DIR)
