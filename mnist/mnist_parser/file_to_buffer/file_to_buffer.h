#ifndef FILE_TO_BUFFER_H_
#define FILE_TO_BUFFER_H_

#include <stdio.h>

/**
 * @brief Puts the contents of a file into a buffer.
 * 
 * @param[out] size The size of the buffer in bytes.
 * @param[in]  file A pointer to the file to be read.
 * 
 * @return A pointer to the buffer containing the file data, or NULL on failure.
 * 
 * @note The buffer is allocated with calloc and must be freed by the caller.
 * @note The buffer is null-terminated for convenience, but the null character is not included in the size.
 */
void* ftbPutFileToBuffer(size_t* size, FILE* file);

#endif