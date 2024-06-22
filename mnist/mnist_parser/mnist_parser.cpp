#include "mnist_parser.h"
#include "file_to_buffer/file_to_buffer.h"

#include <assert.h>
#include <stdlib.h>

MnistParser::MnistParser(const char* images_path, const char* labels_path) {

    images_file_ = fopen(images_path, "rb");
    labels_file_ = fopen(labels_path, "rb");
    // FIXME: throw on error
    assert(images_file_);
    assert(labels_file_);

    images_file_buffer_ = (uint8_t*)ftbPutFileToBuffer(&images_file_buffer_size_, images_file_);
    labels_file_buffer_ = (uint8_t*)ftbPutFileToBuffer(&labels_file_buffer_size_, labels_file_);
    // FIXME: throw on error
    assert(images_file_buffer_);
    assert(labels_file_buffer_);

}

uint32_t MnistParser::ConvertHighEndian(uint8_t* buffer) {
    uint32_t result = 0;
    result |= static_cast<uint32_t>(buffer[0]) << 24;
    result |= static_cast<uint32_t>(buffer[1]) << 16;
    result |= static_cast<uint32_t>(buffer[2]) << 8;
    result |= static_cast<uint32_t>(buffer[3]) << 0;
    return result;
}


const MnistImages MnistParser::GetMnistImages() {
    uint32_t magic_number = ConvertHighEndian(images_file_buffer_ + 0);

    assert(magic_number == 2051);

    uint32_t n_images = ConvertHighEndian(images_file_buffer_ + 4 );
    uint32_t n_rows   = ConvertHighEndian(images_file_buffer_ + 8 );
    uint32_t n_cols   = ConvertHighEndian(images_file_buffer_ + 12);
    uint8_t* buffer   =                  (images_file_buffer_ + 16);

    const MnistImages mnist_images = {
        .n_images = n_images,
        .n_rows   = n_rows,
        .n_cols   = n_cols,
        .buffer   = buffer,
    };

    return mnist_images;
}


const MnistLabels MnistParser::GetMnistLabels() {
    uint32_t magic_number = ConvertHighEndian(labels_file_buffer_ + 0);

    assert(magic_number == 2049);

    uint32_t n_labels = ConvertHighEndian(labels_file_buffer_ + 4);
    uint8_t* buffer   =                  (labels_file_buffer_ + 8);

    const MnistLabels mnist_labels = {
        .n_labels = n_labels,
        .buffer   = buffer,
    };

    return mnist_labels;
}


MnistParser::~MnistParser() {
    fclose(images_file_);
    fclose(labels_file_);

    free(images_file_buffer_);
    free(labels_file_buffer_);
}
