#include "file_to_buffer.h"

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


static ssize_t getFileSize(FILE* file)
{
    struct stat bf = {};
    if (fstat(fileno(file), &bf) == -1)
       return -1;

    return (ssize_t)bf.st_size;
}


static char* putFileToBuffer(const size_t size, FILE* file)
{
    // Size is in BYTES (not the amount of 'char's).
    char* buffer = (char*) calloc(size + 1, 1);
    if (buffer == NULL)
        return NULL;

    size_t sizeRef = fread(buffer, 1, size, file);
    if (sizeRef != size)
    {
        free(buffer);
        return NULL;
    }

    buffer[size] = '\0';
    return buffer;
}


void* ftbPutFileToBuffer(size_t* size, FILE* file)
{
    ssize_t bufferSize = getFileSize(file);
    if (bufferSize == -1)
        return NULL;

    char* buffer = putFileToBuffer(bufferSize, file);

    if (buffer == NULL)
        return NULL;
    buffer[bufferSize] = '\0';

    if (size != NULL)
        *size = bufferSize;

    return buffer;
}
