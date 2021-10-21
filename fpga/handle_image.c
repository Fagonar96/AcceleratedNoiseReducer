#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "image.c"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"


int width, height, channels;

/**
 * Funcion leer una imagen png
 * filepath: ruta de la imagen
 * return: matriz de c con la representacion de la imagen
*/
Image * read_image(Image* image,char *filepath)
{   

    unsigned char *img = stbi_load(filepath, &width, &height, &channels, 0);

    if(img == NULL) {
        printf("Error in loading the image\n");
        exit(1);
    }

    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);

    size_t img_size = width * height * channels;

    int count = 0;
    int* read_data = malloc(width * height * sizeof(int));
    
    for(unsigned char *p = img; p != img + img_size; p+= channels)
    {
        int pixel = (uint8_t)((*p + *(p + 1) + *(p + 2))/3.0);
        read_data[count] = pixel;
        count++;
        //if (count <= 10)
            //printf("Value = %d\n", pixel);
    }
    //printf("Count = %d\n", count);

    count = 0;
    for (int i = 0; i <  width; i++)
    {   
        for (int j = 0; j < height; j++)
        {
            image->data[i][j] = read_data[count];
            count++;
            //if (i == 0 && count <= 10)
                //printf("Value read = %d\n", image->data[i][j]);

        }
    }
    //printf("Count = %d\n", count);

    stbi_image_free(img);
    free(read_data);

    return image;
}

/**
 * Funcion que escribe una imagen en un archivo .png
 * filename: nombre de la imagen a escribir.
 * image: struct de tipo Image con la informacion de la imagen a escribir
*/
void write_image(char *filename, Image *image)
{
    int count = 0;
    int* write_data = malloc(width * height * sizeof(int));

    for (int i = 0; i <  width; i++)
    {   
        for (int j = 0; j < height; j++)
        {
            write_data[count] = image->data[i][j];
            count++;
            //if (i == 0 && count <= 10)
                //printf("Value write= %d\n", image->data[i][j]);
        }
    }
    //printf("Count = %d\n", count);

    int gray_channels = channels == 4 ? 2 : 1; 
    size_t gray_img_size = width * height * gray_channels;
    unsigned char *write_image = malloc(gray_img_size);

    if(write_image == NULL) {
        printf("Unable to allocate memory for the gray image.\n");
        exit(1);
    }

    count = 0;
    for (unsigned char *pg = write_image; pg != write_image + gray_img_size; pg += gray_channels)
    {
        *pg = (uint8_t) write_data[count];
        count++;
        //if (count <= 10)
            //printf("Value = %d\n", *pg);
    }
    //printf("Count = %d\n", count);

    stbi_write_png(filename, width, height, gray_channels, write_image, width * gray_channels);

    free(write_image);
    free(write_data);
}