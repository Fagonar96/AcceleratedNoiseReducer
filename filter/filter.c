#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include "handle_image.c"

#define WINDOW_SIZE 1
#define NEIGHBORHOOD_SIZE ((WINDOW_SIZE * 2 + 1) * (WINDOW_SIZE * 2 + 1))
struct stat st = {0};


/**
 * This function captures the output of a system command
 * command: 
*/
double execute(const char *command)
{   
    char buffer[16];
    int buffer_size = sizeof(buffer);

    //printf("%s\n", command);

    FILE *pipe = popen(command, "r");
    if (pipe == NULL)
    {
        printf("Error popen failed!");
        exit(1);
    }
    
    fgets(buffer, 16, pipe);
    char *ptr;

    double result = strtod(buffer, &ptr);

    //char *result = (char *) malloc(buffer_size+1);
    //strcpy(result, buffer);    

    pclose(pipe);
    
    return result;
}


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
double filter(Image *input_image, Image *filtered_image, int window_size, int file_number, 
            FILE *fptr_time, FILE *fptr_mem, FILE *fptr_cpu)
{
    // Variables to measure time
    double start_time, frame_time;
    // Start the frame execution time
    start_time = omp_get_wtime();


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
            temp = round(temp / NEIGHBORHOOD_SIZE);
            int result = (int) temp;
            filtered_image->data[i][j] = result;
        }
    }

    int pid = getpid();
    //printf("Frame PID:%d\n", pid);

    // Stop the frame execution time
    frame_time = omp_get_wtime() - start_time;
    // Write frame time to time file
    fprintf(fptr_time, "Frame %d = %f s\n", file_number, frame_time);

    char *mem_command;
    asprintf(&mem_command, "ps -o rss= %d", pid);
    double memory_usage = execute(mem_command) / 1024;
    fprintf(fptr_mem, "Frame %d = %f MB\n", file_number, memory_usage);

    char* cpu_command;
    asprintf(&cpu_command, "ps -o pcpu= %d", pid);
    double cpu_usage = execute(cpu_command);
    if (cpu_usage >= 100)
        cpu_usage = 100;
    fprintf(fptr_cpu, "Frame %d = %.1f %\n", file_number, cpu_usage);

    return frame_time;
}


/** 
 * This function applies batch processing of the files containing the input 
 * images and writes the filtered images.
 * input_directory: directory path containing the input images.
 * file_amount: the amount of input images to be processed.
*/
int process_files(const char *input_directory, int file_amount, int batch_amount)
{
    // Creates filtered frames folder if it doesnt exist
    if (stat("../filtered_serial", &st) == -1)
    {
        mkdir("../filtered_serial", 0700);
    }
    // Remove measurement files if they exist
    remove("../filtered_serial/time.txt");
    remove("../filtered_serial/memory.txt");
    remove("../filtered_serial/cpu.txt");

    // Set memory space for input frames
    Image *input_images = (Image*) malloc(batch_amount * sizeof(*input_images));

    // Set memory space for output frames
    Image *filtered_images = (Image*) malloc(batch_amount * sizeof(*filtered_images));
    
    // Variable to store full execution time
    double full_time = 0;

    // File for runtime measurements
    FILE *fptr_time = fopen("../filtered_serial/time.txt", "a");
    fprintf(fptr_time,"Frame Filtering Runtime Measurements\n\n");

    // File for memory usage measurements
    FILE *fptr_mem = fopen("../filtered_serial/memory.txt", "a");
    fprintf(fptr_mem,"Frame Filtering Memory Usage Measurements\n\n");

    // File for cpu usage measurements
    FILE *fptr_cpu = fopen("../filtered_serial/cpu.txt", "a");
    fprintf(fptr_cpu,"Frame Filtering CPU Usage Measurements\n\n");

    int batches = file_amount/batch_amount;
    for (int batches_c = 0; batches_c < batches; batches_c++)
    {
        // Load and read the batch of frames
        for (int files_c = 0; files_c < batch_amount; files_c++)
        {
            int file_number = files_c + batches_c * batch_amount;
            char *filename;
            asprintf(&filename, "%s/frame%d.png", input_directory, file_number);
            printf("Frame %d loaded.\n", file_number);
            Image *image = read_image(&input_images[files_c], filename);
        }
        
        // Filter and process the batch of frames
        for (int filter_c = 0; filter_c < batch_amount; filter_c++)
        {   
            int file_number = filter_c + batches_c * batch_amount;
            // Call the filter function
            double frame_time = filter(&input_images[filter_c], &filtered_images[filter_c],
                                       1, file_number, fptr_time, fptr_mem, fptr_cpu);
            // Adds frame time to full time
            full_time += frame_time;
            printf("Frame %d filtered.\n", file_number);
        }

        // Save and write the batch of frames
        for (int file_write_c = 0; file_write_c < batch_amount; file_write_c++)
        {
            int file_number=file_write_c+batches_c*batch_amount;
            char *filename;
            asprintf(&filename, "../filtered_serial/frame%d.png", file_number);
            write_image(filename, &filtered_images[file_write_c]);
            printf("Frame %d saved.\n", file_number);
        }
    }

    //printf("Execution time: %f\n", full_time);
    fprintf(fptr_time,"\nTotal Runtime = %f s", full_time);

    // Close the files
    fclose(fptr_time);
    fclose(fptr_mem);

    // Free up memory space
    free(filtered_images);
    free(input_images);

    return 0;
}

// ./median_filter ../../frame 5
int main(int argc, char *argv[])
{
    // Print the arguments received
    for (int i = 0; i < argc; i++)
    {
        printf("argv[%d]: %s\n", i, argv[i]);
    }
    
    // Check the number of arguments received
    if (argc != 4)
    {
        printf("Error in arguments\n");
        return 1;
    }

    // Obtain the arguments
    const char *input_directory_arg = argv[1];
    const char *num_frames_arg = argv[2];
    const char *num_batch_arg = argv[3];

    int num_frames = atol(num_frames_arg);
    int num_batch = atol(num_batch_arg);

    printf("Input directory: %s\n", input_directory_arg);

    if (num_frames%num_batch == 0 & num_frames != 0)
    {
        process_files(input_directory_arg, num_frames, num_batch);
    }
    else{
        printf("Number of files to process must be multiple of %d \n",num_batch);
    }

    return 0;
}
