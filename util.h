#ifndef UTIL_H
#define UTIL_H

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// Read a file into an allocated buffer
char* read_file(const char* filename);

// Check and handle OpenCL errors
void check_error(cl_int err, const char* operation);

#endif // UTIL_H