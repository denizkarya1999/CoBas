#include <stdint.h>

#include "MLX90642.h"

/*
 * Thin C ABI used by Python's ctypes layer.  The shim fixes the sensor address
 * and exposes the few frame operations and geometry values the app needs.  It
 * owns no sensor state and allocates no frame storage.
 */
#define FRAME_PIXELS MLX90642_TOTAL_NUMBER_OF_PIXELS

/* Initialize the device at the address used by the single-camera Python app. */
int MLX90642_PythonInit(void)
{
    /* Preserve the native driver's status value for the Python caller. */
    return MLX90642_Init(SA_90642_DEFAULT);
}

/*
 * Copy one calculated image into caller-owned storage.  ctypes must provide a
 * writable array of at least FRAME_PIXELS uint16_t elements; this function
 * neither allocates nor retains that array.
 */
int MLX90642_PythonReadFrame(uint16_t *frame)
{
    /* Reject a null foreign-function pointer before it reaches the I2C driver. */
    if (frame == 0) {
        return -MLX90642_INVAL_VAL_ERR;
    }

    /* GetImage returns zero on success and a negative driver error on failure. */
    return MLX90642_GetImage(SA_90642_DEFAULT, frame);
}

/*
 * Wait for a transition to a new read window, rather than merely testing
 * whether a window is open.  Requiring an observed close prevents an already
 * open/current window from being mistaken for the next one.
 */
int MLX90642_PythonWaitForNextFrame(uint16_t max_polls)
{
    /* A closed window must be observed before a later open one is "next". */
    int saw_closed_window = 0;

    for (uint16_t poll = 0; poll < max_polls; poll++) {
        int status = MLX90642_IsReadWindowOpen(SA_90642_DEFAULT);

        /* Do not turn an I2C/driver failure into a timeout. */
        if (status < 0) {
            return status;
        }

        if (status == MLX90642_NO) {
            saw_closed_window = 1;
        } else if (saw_closed_window) {
            return 0;
        }

        /* Bound bus traffic and make max_polls define the timeout budget. */
        MLX90642_Wait_ms(MLX90642_POLL_TIME_MS);
    }

    return -MLX90642_TIMEOUT_ERR;
}

/* Geometry helpers expose scalar values that ctypes can query safely. */
int MLX90642_PythonFrameWidth(void)
{
    return 32;
}

int MLX90642_PythonFrameHeight(void)
{
    /* Derive 24 rows from the driver's 768-word frame-size contract. */
    return FRAME_PIXELS / 32;
}
