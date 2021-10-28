#include <stdio.h>
#include <string.h>
#include <png.h>
#include <stdlib.h>
#include "image.c"

/**
 * Funcion leer una imagen png
 * filepath: ruta de la imagen
 * return: matriz de c con la representacion de la imagen
*/
Image * read_image(Image* image,char *filepath)
{
    FILE *pFile = fopen(filepath, "rb");
    if (!pFile)
    {
        printf("Error al leer el archivo %s\n", filepath);
        exit(1);
    }
    // Se leen los primeros 8 bits del archivo para
    // verificar que sea una imagen en formato png
    int len = 8;                  // Largo del buffer
    char header[len];             // Buffer
    fread(header, 1, len, pFile); // Lectura de los primeros 8 bits
    int is_png = !png_sig_cmp(header, 0, len);
    if (!is_png)
    {
        printf("Archivo %s no es una imagen en formato png\n", filepath);
        fclose(pFile);
        exit(1);
    }

    // Creacion de la estructura de lectura
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("Error al obtener la informacion del archivo %s\n", filepath);
        fclose(pFile);
        exit(1);
    }

    // Lectura de la informacion de la imagen
    png_init_io(png_ptr, pFile);
    png_set_sig_bytes(png_ptr, len); // Se indica que se han leido 8 bits
    png_read_info(png_ptr, info_ptr);
    int width = png_get_image_width(png_ptr, info_ptr);   // Ancho
    int height = png_get_image_height(png_ptr, info_ptr); // Largo
    int channels = png_get_channels(png_ptr, info_ptr);   // Canales
    //printf("Channels = %d\n", channels);
    int number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    // Lectura de los datos
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("Error durante la lectura de los pixeles\n");
        fclose(pFile);
        exit(1);
    }

    // Memoria para almacenar los pixeles de la imagen
    png_bytep *row_pointers = (png_bytep *)malloc(sizeof(png_bytep) * height);
    for (int i = 0; i < height; i++)
    {
        row_pointers[i] = (png_byte *)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    // int **data = createMatrix(width, height);

    // CREATE_IMAGE(image)

    // Lectura de los pixeles
    png_read_image(png_ptr, row_pointers);
    for (int y = 0; y < height; y++)
    {
        png_byte *row = row_pointers[y];
        for (int x = 0; x < width; x++)
        {
            png_byte *pixel = &(row[x * channels]);
            // Lectura del pixel, se guarda en la posicion correspondiente en data
            image->data[x][y] = pixel[0];
        }
    }

    // Limpieza de memoria
    fclose(pFile);
    for (int i = 0; i < height; i++)
        free(row_pointers[i]);
    free(row_pointers);

    return image;
}

/**
 * Funcion que escribe una imagen en un archivo .png
 * image: struct de tipo Image con la informacion de la imagen a escribir
*/
void write_image(char *filename, Image *image)
{
    FILE *fp = NULL;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    png_bytep row = NULL;

    // Apertura del archivo para escritura
    fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        fprintf(stderr, "Error al crear el archivo %s para escritura\n", filename);
        exit(1);
    }

    // Inicializacion de estructura de escritura
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL)
    {
        fprintf(stderr, "Could not allocate write struct\n");
        exit(1);
    }

    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        fprintf(stderr, "Could not allocate info struct\n");
        exit(1);
    }

    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        fprintf(stderr, "Error during png creation\n");
        exit(1);
    }

    png_init_io(png_ptr, fp);

    // Write header (8 bit colour depth)
    png_set_IHDR(png_ptr, info_ptr, IMAGE_M, IMAGE_N,
                 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    // Allocate memory for one row (1 byte per pixel - RGB)
    row = (png_bytep)malloc(IMAGE_M * sizeof(png_byte));

    // Write image data
    for (int y = 0; y < IMAGE_N; y++)
    {
        for (int x = 0; x < IMAGE_M; x++)
        {
            png_byte *pixel = &(row[x]);
            *pixel = image->data[x][y];
        }
        png_write_row(png_ptr, row);
    }

    // End write
    png_write_end(png_ptr, NULL);

    // Limpieza de memoria
    fclose(fp);
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    free(row);
}
