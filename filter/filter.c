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


//  method returns Nth power of A
double nthRoot(int A, int N)
{
    // initially guessing a random number between
    // 0 and 9
    double xPre = rand() % 10;
 
    //  smaller eps, denotes more accuracy
    double eps = 1e-3;
 
    // initializing difference between two
    // roots by INT_MAX
    double delX = INT_MAX;
 
    //  xK denotes current value of x
    double xK;
 
    //  loop until we reach desired accuracy
    while (delX > eps)
    {
        //  calculating current value from previous
        // value by newton's method
        xK = ((N - 1.0) * xPre +
              (double)A/pow(xPre, N-1)) / (double)N;
        delX = abs(xK - xPre);
        xPre = xK;
    }
 
    return xK;
}

/** 
 * This function applies the geometric avergae filter to an input image
*/
void filter(Image *input_image,Image *filtered_image , int window_size)
{

    for (int i = 1; i < IMAGE_M - 1; i++)
    {
        for (int j = 1; j < IMAGE_N - 1; j++)
        {
            int x_start = i - window_size;
            int x_end = i + window_size;
            int y_start = j - window_size;
            int y_end = j + window_size;
            
            double temp = 1;

            for (int x = x_start; x <= x_end; x++)
            {
                for (int y = y_start; y <= y_end; y++)
                {
                    int pixel = input_image->data[x][y];
                    //printf("Input Pixel  = %d\n", pixel);
                    //temp = temp * pixel;
                    temp = temp + pixel;
                }
            }
            //printf("Output Pixel BEFORE POW= %f\n", temp);
            //temp = round(pow(temp, 1.0/9));
            temp = round(temp/9);
            //printf("Output Pixel= %f\n", temp);

            int result = (int) temp;
            //printf("Cast output pixel = %d\n", result);

            filtered_image->data[i][j] = result;

            //printf("Next neighborhood\n");
        }
    }
}

int process_files(const char *input_directory, int file_amount)
{
    // Creates filtered frames folder if it doesnt exist
    if (stat("../filtered", &st) == -1)
    {
        mkdir("../filtered", 0700);
    }
    
    // Reserve memory for input frames
    Image (*input_images)[PARALLEL_FILES_TO_LOAD] = malloc(sizeof(*input_images));
    for (int i = 0; i < PARALLEL_FILES_TO_LOAD; i++)
    {
        Image * imageptr;
        imageptr= &(*input_images)[i];
        RESET_IMAGE(  imageptr )
    }

    // Reserve memory for output frames
    Image (*filtered_images)[PARALLEL_FILES_TO_LOAD] = malloc(sizeof(*filtered_images));
    for (int i = 0; i < PARALLEL_FILES_TO_LOAD; i++)
    {
        Image * imageptr;
        imageptr= &(*filtered_images)[i];
        RESET_IMAGE(imageptr)
    }
    
    // Variable for full execution time
    double full_time = 0;

    int batches = file_amount/PARALLEL_FILES_TO_LOAD;
    for (int batches_c = 0; batches_c < batches; batches_c++)
    {
        // Load frames
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
        
        // Process frames
        for (int filter_c = 0; filter_c < PARALLEL_FILES_TO_LOAD; filter_c++)
        {
            
            // Call filter
            filter(&(*input_images)[filter_c],&(*filtered_images)[filter_c] , 1);
            printf("Frame %d procesado.\n", filter_c);
        }


        run_time = omp_get_wtime() - start_time;
        full_time += run_time;

        // Save frames
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

    free(filtered_images);
    free(input_images);

    return 0;
}

// ./median_filter ../../frame 5
int main(int argc, char *argv[])
{
    // printf("ARGC: %d\n", argc);

    // TODO: validar argumentos
    int i;
    for (i = 0; i < argc; i++)
    {
        printf("argv[%d]: %s\n", i, argv[i]);
    }

    if (argc != 3)
    {
        printf("error in arguments\n");
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
