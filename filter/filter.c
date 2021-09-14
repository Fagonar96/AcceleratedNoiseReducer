#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include "handle_image.c"

#define WINDOW_SIZE 1
#define NEIGHBORHOOD_SIZE ((WINDOW_SIZE * 2 + 1) * (WINDOW_SIZE * 2 + 1))
#define PARALLEL_FILES_TO_LOAD 1
struct stat st = {0};


/** 
 * This function applies the filter to an input image and returns a filtered image
 * input_image: struct with the data of the input image with noise
 * filtered_image: struct with data of the filtered image with reduced noise
 * window_size: size of the neighborhood window
 *              1 -> 3 x 3 window
 *              2 -> 5 x 5 window
 *              3 -> 7 x 7 window
 *              4 -> 9 x 9 window 
*/
void filter(Image *input_image, Image *filtered_image, int window_size)
{

    // Double loop to travel the frame matrix
    for (int i = 1; i < IMAGE_M - 1; i++)
    {
        for (int j = 1; j < IMAGE_N - 1; j++)
        {   
            // Neighboirhood boundaries
            int x_start = i - window_size;
            int x_end = i + window_size;
            int y_start = j - window_size;
            int y_end = j + window_size;

            // Initial value of filtered pixel
            double temp = 0;

            // Double loop to travel the temporary nieghborhood
            for (int x = x_start; x <= x_end; x++)
            {
                for (int y = y_start; y <= y_end; y++)
                {
                    // Average filter
                    int pixel = input_image->data[x][y];
                    temp = temp + pixel;
                }
            }
            // Average filter
            temp = round(temp/NEIGHBORHOOD_SIZE);
            int result = (int) temp;
            filtered_image->data[i][j] = result;
        }
    }
}


/** 
 * This function applies batch processing of the files containing the input 
 * images and writes the filtered images.
 * input_directory: directory path containing the input images.
 * file_amount: the amount of input images to be processed.
*/
int process_files(const char *input_directory, int file_amount)
{
    // Creates filtered frames folder if it doesnt exist
    if (stat("../filtered", &st) == -1)
    {
        mkdir("../filtered", 0700);
    }
    
    // Set memory space for input frames
    Image (*input_images)[PARALLEL_FILES_TO_LOAD] = malloc(sizeof(*input_images));
    for (int i = 0; i < PARALLEL_FILES_TO_LOAD; i++)
    {
        Image * imageptr;
        imageptr= &(*input_images)[i];
        RESET_IMAGE(  imageptr )
    }

    // Set memory space for output frames
    Image (*filtered_images)[PARALLEL_FILES_TO_LOAD] = malloc(sizeof(*filtered_images));
    for (int i = 0; i < PARALLEL_FILES_TO_LOAD; i++)
    {
        Image * imageptr;
        imageptr= &(*filtered_images)[i];
        RESET_IMAGE(imageptr)
    }
    
    // Variable to store full execution time
    double full_time = 0;

    int batches = file_amount/PARALLEL_FILES_TO_LOAD;
    for (int batches_c = 0; batches_c < batches; batches_c++)
    {
        // Load and read the batch of frames
        #pragma omp parallel for
        for (int files_c = 0; files_c < PARALLEL_FILES_TO_LOAD; files_c++)
        {
            int file_numer=files_c+batches_c*PARALLEL_FILES_TO_LOAD;
            char *filename;
            asprintf(&filename, "%s/frame%d.png", input_directory, file_numer);
            printf("filename: %s\n", filename);
            Image *image = read_image( &(*input_images)[files_c],filename);
        }

        double start_time, run_time;
        start_time = omp_get_wtime();
        
        // Process the batch of frames
        for (int filter_c = 0; filter_c < PARALLEL_FILES_TO_LOAD; filter_c++)
        {
            
            // Call the filter function
            filter(&(*input_images)[filter_c],&(*filtered_images)[filter_c] , 1);
            printf("Frame %d procesado.\n", filter_c);
        }


        run_time = omp_get_wtime() - start_time;
        full_time += run_time;

        // Save and write the batch of frames
        #pragma omp parallel for
        for (int file_write_c = 0; file_write_c < PARALLEL_FILES_TO_LOAD; file_write_c++)
        {
            int file_numer=file_write_c+batches_c*PARALLEL_FILES_TO_LOAD;
            char *filename;
            asprintf(&filename, "../filtered/frame%d.png", file_numer);
            write_image(filename, &(*filtered_images)[file_write_c]);
            printf("Frame %d guardado.\n", file_write_c);
        }
    }

    printf("Execution time: %f\n", full_time);

    // Free up memory space
    free(filtered_images);
    free(input_images);

    return 0;
}

// ./median_filter ../../frame 5
int main(int argc, char *argv[])
{
    int i;
    for (i = 0; i < argc; i++)
    {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

    if (argc != 3)
    {
        printf("Error in arguments\n");
        return 1;
    }

    const char *input_directory_arg = argv[1];
    const char *num_arg = argv[2];
    int num = atol(num_arg);

    printf("input_directory_arg: %s\n", input_directory_arg);

    if (num%PARALLEL_FILES_TO_LOAD==0 & num!=0)
    {
        process_files(input_directory_arg, num);
    }
    else{
        printf("Number of files to process must be multiple of %d \n",PARALLEL_FILES_TO_LOAD);
    }

    return 0;
}
