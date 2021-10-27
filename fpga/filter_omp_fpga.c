#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include "handle_image.c"

#define WINDOW_SIZE 1
#define NEIGHBORHOOD_SIZE ((WINDOW_SIZE * 2 + 1) * (WINDOW_SIZE * 2 + 1))
struct stat st = {0};

/**
 * This function captures the output of a system command and return 
 * the output converted as a double.
 * command: string of the command to be executed
*/
double execute(const char *command)
{   
    // Declare a buffer to store command output
    char buffer[16]; int buffer_size = sizeof(buffer);

    // Print the command to be executed
    //printf("%s\n", command);

    // Open a pipe for the command process
    FILE *pipe = popen(command, "r");
    if (pipe == NULL)
    {
        printf("Error popen failed!");
        exit(1);
    }
    
    // Get the command output from pipe
    fgets(buffer, 16, pipe);
    
    // Convert command output to double
    char *ptr; double result = strtod(buffer, &ptr);

    // Convert the command output to string
    //char *result = (char *) malloc(buffer_size+1);
    //strcpy(result, buffer);    

    // Close the pipe of the command process
    pclose(pipe);
    
    // Return the command output
    return result;
}


/** 
 * This function applies the filter to an input image and returns a filtered image
 * and its frame filtering runtime.
 * input_image: struct with the data of the input image with noise
 * filtered_image: struct with data of the filtered image with reduced noise
 * window_size: size of the neighborhood window
 *              1 -> 3 x 3 window
 *              2 -> 5 x 5 window
 *              3 -> 7 x 7 window
 *              4 -> 9 x 9 window
 * frame_time: frame filtering run time
 * memory_usage: frame filtering memory usage
 * file_number: frame file number
*/
void filter(Image *input_image, Image *filtered_image, int window_size, 
            double* frame_time, double* memory_usage, double* cpu_usage, int file_number)
{
    // Variables to measure time
    double start_time;
    // Start the frame runtime
    start_time = omp_get_wtime();

    // Double loop to travel the frame matrix
    #pragma omp parallel for collapse(2)
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

    // Get the process ID
    int pid = getpid();
    //printf("Frame PID:%d\n", pid);

    // Stop the frame execution time
    *frame_time = omp_get_wtime() - start_time;

    // Create the memory usage command with process ID
    char *mem_command; asprintf(&mem_command, "ps -o rss= %d", pid);
    // Run the command and get its memory usage output
    *memory_usage = execute(mem_command) / 1024;

    // Create the cpu utilization command with process ID
    char* cpu_command; asprintf(&cpu_command, "ps -o pcpu= %d", pid);
    // Run the command and get its cpu utilization output
    *cpu_usage = execute(cpu_command);
}


/** 
 * This function applies batch processing of the files containing the input 
 * images and writes the filtered images.
 * input_directory: directory path containing the input images.
 * file_amount: the amount of input images to be processed.
 * batch_amount: the amount of batches of input images.
*/
int process_files(const char *input_directory, int file_amount, int batch_amount)
{
    // Creates filtered frames folder if it doesnt exist
    if (stat("filtered_omp_fpga", &st) == -1)
    {
        mkdir("filtered_omp_fpga", 0700);
    }
    system("chmod +rwx filtered_omp_fpga/");
    // Remove measurement files if they exist
    remove("filtered_omp_fpga/time.txt");
    remove("filtered_omp_fpga/memory.txt");
    remove("filtered_omp_fpga/cpu.txt");

    // Open file for runtime measurements
    FILE *fptr_time = fopen("filtered_omp_fpga/time.txt", "a");
    fprintf(fptr_time,"Frame Filtering Runtime Measurements\n\n");

    // Open file for memory usage measurements
    FILE *fptr_mem = fopen("filtered_omp_fpga/memory.txt", "a");
    fprintf(fptr_mem,"Frame Filtering Memory Usage Measurements\n\n");

    // Open file for cpu usage measurements
    FILE *fptr_cpu = fopen("filtered_omp_fpga/cpu.txt", "a");
    fprintf(fptr_cpu,"Frame Filtering CPU Usage Measurements\n\n");
    
    // Set memory space for input frames
    Image *input_images = (Image*) malloc(batch_amount * sizeof(*input_images));

    // Set memory space for output frames
    Image *filtered_images = (Image*) malloc(batch_amount * sizeof(*filtered_images));

    // Variable for full runtime
    double full_time = 0;
    // Variable for average memory usage
    double full_memory = 0;

    // Travel the amount of batches
    int batches = file_amount/batch_amount;
    for (int batches_c = 0; batches_c < batches; batches_c++)
    {
        // Load and read the batch of frames
        #pragma omp parallel for
        for (int files_c = 0; files_c < batch_amount; files_c++)
        {
            int file_number = files_c + batches_c * batch_amount;
            char *filename;
            asprintf(&filename, "%s/frame%d.png", input_directory, file_number);
            printf("Frame %d loaded.\n", file_number);
            Image *image = read_image(&input_images[files_c], filename);
        }

        // Variables for frame time, memory usage and cpu utilization
        double frame_time, frame_memory, frame_cpu;

        // Filter and process the batch of frames
        //#pragma omp parallel for
        for (int filter_c = 0; filter_c < batch_amount; filter_c++)
        {   
            // Variable for file number
            int file_number = filter_c + batches_c * batch_amount;
            // Call the filter function
            filter(&input_images[filter_c], &filtered_images[filter_c], WINDOW_SIZE,
                   &frame_time, &frame_memory, &frame_cpu, file_number);

            // Write frame time to time file
            fprintf(fptr_time, "Frame %d = %f s\n", file_number, frame_time);
            // Write the memory usage result to the memory file
            fprintf(fptr_mem, "Frame %d = %f MB\n", file_number, frame_memory);
            // Write the cpu utilization result to the cpu file
            if (frame_cpu >= 100 || frame_cpu == 0) frame_cpu = 100;
            fprintf(fptr_cpu, "Frame %d = %.1f %\n", file_number, frame_cpu);

            // Adds frame time to full time
            full_time += frame_time;
            // Adds memory_usage to full memory
            full_memory += frame_memory;

            printf("Frame %d filtered.\n", file_number);
        }

        // Save and write the batch of frames
        #pragma omp parallel for
        for (int file_write_c = 0; file_write_c < batch_amount; file_write_c++)
        {
            int file_number=file_write_c+batches_c*batch_amount;
            char *filename;
            asprintf(&filename, "filtered_omp_fpga/frame%d.png", file_number);
            write_image(filename, &input_images[file_write_c]);
            printf("Frame %d saved.\n", file_number);
        }
    }

    // Write the full runtime of the process
    fprintf(fptr_time,"\nTotal Runtime = %f s", full_time);
    //printf("Execution time: %f\n", full_time);

    // Write the average memory usage of the process
    full_memory = full_memory / file_amount;
    fprintf(fptr_mem, "\nAverage Memory Usage = %f MB", full_memory);
    //printf("Average memory usage: %f\n", full_memory);

    // Close the files
    fclose(fptr_time);
    fclose(fptr_mem);
    fclose(fptr_cpu);

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

    // Convert the number arguments into integers
    int num_frames = atol(num_frames_arg);
    int num_batch = atol(num_batch_arg);

    // Print the input directory path
    printf("Input directory: %s\n", input_directory_arg);

    // Validate that the number of batches is a multiple of the number of frames
    if (num_frames%num_batch == 0 & num_frames != 0)
    {
        process_files(input_directory_arg, num_frames, num_batch);
    }
    else
    {
        printf("Number of files to process must be multiple of %d \n",num_batch);
    }

    return 0;
}



