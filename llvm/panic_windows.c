#include <stdlib.h>
#include <signal.h>

extern void gopanic(char* cmsg);

void SignalHandler(int signal)
{
    if (signal == SIGABRT)
    {
        // Trigger the panic
        gopanic("abort() has been called");
    }
}

void init_panic_handler() {
    signal(SIGABRT, SignalHandler);
}