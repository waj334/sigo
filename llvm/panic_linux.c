#include <stdlib.h>
#include <stdio.h>

#define MAX_MSG_LENGTH 1024

extern void gopanic(char* cmsg);

void __assert_fail(const char* expr, const char *filename, unsigned int line, const char *assert_func)
{
    // Allocate memory for error string
    char* buf = (char*)malloc(MAX_MSG_LENGTH);

    // Format the message
    snprintf( buf, MAX_MSG_LENGTH, "%s:%d:4: error: assertion \"%s\" failed in function %s\n",
                                filename, line, expr, assert_func );

    // Trigger the panic
    gopanic(buf);
}