// #define IMAGE_N 3840
// #define IMAGE_M 2160

#ifndef IMAGE_N
#define IMAGE_N 2160 //COLS
#endif

#ifndef IMAGE_M
#define IMAGE_M 3840 //ROWS
#endif

#define FREE_IMAGE(var_name)        free(var_name);
#define CREATE_IMAGE(var_name)      Image *var_name = malloc(sizeof(Image)); memset(var_name->data, 0, IMAGE_N *IMAGE_M * sizeof(int));
#define MALLOC_IMAGE(image_pointer) image_pointer = malloc(sizeof(Image)); memset(image_pointer->data, 0, IMAGE_N *IMAGE_M * sizeof(int));
#define RESET_IMAGE(image_pointer)  memset(image_pointer->data, 0, IMAGE_N *IMAGE_M * sizeof(int));

typedef struct
{
    int data[IMAGE_M][IMAGE_N];
} Image;


