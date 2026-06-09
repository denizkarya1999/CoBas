#include <stdint.h>

#include "MLX90642.h"

#define FRAME_PIXELS MLX90642_TOTAL_NUMBER_OF_PIXELS

int MLX90642_PythonInit(void)
{
    return MLX90642_Init(SA_90642_DEFAULT);
}

int MLX90642_PythonReadFrame(uint16_t *frame)
{
    if (frame == 0) {
        return -MLX90642_INVAL_VAL_ERR;
    }

    return MLX90642_GetImage(SA_90642_DEFAULT, frame);
}

int MLX90642_PythonWaitForNextFrame(uint16_t max_polls)
{
    int saw_closed_window = 0;

    for (uint16_t poll = 0; poll < max_polls; poll++) {
        int status = MLX90642_IsReadWindowOpen(SA_90642_DEFAULT);

        if (status < 0) {
            return status;
        }

        if (status == MLX90642_NO) {
            saw_closed_window = 1;
        } else if (saw_closed_window) {
            return 0;
        }

        MLX90642_Wait_ms(MLX90642_POLL_TIME_MS);
    }

    return -MLX90642_TIMEOUT_ERR;
}

int MLX90642_PythonFrameWidth(void)
{
    return 32;
}

int MLX90642_PythonFrameHeight(void)
{
    return FRAME_PIXELS / 32;
}
