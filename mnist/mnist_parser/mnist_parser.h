#ifndef MNIST_PARSER_H_
#define MNIST_PARSER_H_

#include <stdio.h>
#include <cinttypes>

struct MnistImages {
    uint32_t n_images;
    uint32_t n_rows;
    uint32_t n_cols;
    uint8_t* buffer;
};

struct MnistLabels {
    uint32_t n_labels;
    uint8_t* buffer;
};

class MnistParser {
    public:
        MnistParser(const char* images_path, const char* labels_path);
        ~MnistParser();
        const MnistImages GetMnistImages();
        const MnistLabels GetMnistLabels();

    private:
        // file_to_buffer is in C, so using C-style here.
        FILE* images_file_;
        FILE* labels_file_;

        uint8_t*    images_file_buffer_;
        std::size_t images_file_buffer_size_;
        uint8_t*    labels_file_buffer_;
        std::size_t labels_file_buffer_size_;

        uint32_t ConvertHighEndian(uint8_t* buffer);
};

#endif // MNIST_PARSER_H_
