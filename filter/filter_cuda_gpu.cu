#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include "handle_image_cuda.c"

#define WINDOW_SIZE 1
#define NEIGHBORHOOD_SIZE ((WINDOW_SIZE * 2 + 1) * (WINDOW_SIZE * 2 + 1))
struct stat st = {0};

// Frame matrix dimensions
int image[IMAGE_M][IMAGE_N];
int filte[IMAGE_M][IMAGE_N];

// CUDA Kernel definition
__global__ void ImageFilter(int A[IMAGE_M][IMAGE_N], int B[IMAGE_M][IMAGE_N], int window_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column address
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row address

    if (i >= 1 && i < IMAGE_M - 1 && j >=1 && j < IMAGE_N - 1)
    {   
        // Initial value of filtered pixel
        double temp = 0;

        // First neighborhood row
        temp += A[i-1][j-1];
        temp += A[i][j-1];
        temp += A[i+1][j-1];

        // Second neighborhood row
        temp += A[i-1][j];
        temp += A[i][j];
        temp += A[i+1][j];

        // Third neighborhood row
        temp += A[i-1][j+1];
        temp += A[i][j+1];
        temp += A[i+1][j+1];

        // Average filter
        temp = round(temp/NEIGHBORHOOD_SIZE);
        int result = (int) temp;
        B[i][j] = result;
    }
}

/**
 * This function captures the output of a system command and return 
 * the output converted as a string.
 * command: string of the command to be executed
*/
char* execute(const char *command)
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
    //char *ptr; double result = strtod(buffer, &ptr);

    // Convert command output to string
    char *result = (char *) malloc(buffer_size+1);
    strcpy(result, buffer);    

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
 * file_number: frame number
 * fptr_time: file pointer to handle the measures of filter runtime
 * fptr_mem: file pointer to handle the measures of filter memory usage
 * fptr_cpu: file pointer to handle the measure of filter CPU usage
*/
double filter(Image *input_image, Image *filtered_image, int window_size, int file_number,
              FILE *fptr_time, FILE *fptr_mem, FILE *fptr_gpu, int mem_int) 
{ 
    // Set memory space for the input and output images
    const int m = IMAGE_M; const int n = IMAGE_N;
    const int size = m * n * sizeof(int);
    memset(image, 0, size);
    memset(filte, 0, size);

    // Copy the input image data to the structure
    //printf("Image = \n");
    for (int i = 1; i < m - 1; i++)
    {
        for (int j = 1; j < n - 1; j++)
        {   
            image[i][j] = input_image->data[i][j];
            //printf("%d ", image[i][j]);
        }
        //printf("\n");
    }

    // Allocate image matrix arrays in device memory
    int (*pImage)[n], (*pFilte)[n];
    cudaMalloc((void**)&pImage, size);
    cudaMalloc((void**)&pFilte, size);

    // Copy matrix arrays from host memory to device memory
    cudaMemcpy(pImage, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(pFilte, filte, size, cudaMemcpyHostToDevice);

    // Define the number of block and threads per block of the device
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(m / threadsPerBlock.x, n / threadsPerBlock.y);
    //printf("numBlocks.x = %d, numBlocks.y = %d\n", numBlocks.x, numBlocks.y);

    // Variables to measure time
    float irun_time;
    cudaEvent_t istart, iend;

     // Start the frame execution time
    cudaEventCreate(&istart);
    cudaEventCreate(&iend);
    cudaEventRecord(istart, 0);

    // Call the kernel definition
    ImageFilter<<<numBlocks, threadsPerBlock>>>(pImage, pFilte, window_size);

    // Stop the frame execution time
    cudaEventRecord(iend, 0);
	cudaEventSynchronize(iend);
	cudaEventElapsedTime(&irun_time, istart, iend);
    
    // Write frame time to time file
    fprintf(fptr_time, "Frame %d = %f s\n", file_number, irun_time/1000);
    //printf("Frame time = %f\n", irun_time/1000);

    // Create the GPU memory usage query command
    char *mem_command = "nvidia-smi --query-gpu=memory.used --format=csv | tail -1";
    // Run the command and get its GPU memory usage output
    char *mem_result = execute(mem_command);
    int memory_usage = atoi(mem_result) - mem_int;
    // Write the GPU memory usage result to the memory file
    fprintf(fptr_mem, "Frame %d = %d MB\n", file_number, memory_usage);
    
    // Create the GPU utilization query command
    char *gpu_command = "nvidia-smi --query-gpu=utilization.gpu --format=csv | tail -1";
    // Run the command and get its GPU utilization output
    char *gpu_usage = execute(gpu_command);
    // Write the GPU utilization result to the GPU file
    fprintf(fptr_gpu, "Frame %d = %s", file_number, gpu_usage);
    
    // Copy result from device memory to host memory
    cudaMemcpy(filte, pFilte, (m*n)*sizeof(int), cudaMemcpyDeviceToHost);
    
    // Copy the filtered data structure to the ouput image
    //printf("Filtered = \n");
    for (int i = 1; i < m - 1; i++)
    {
        for (int j = 1; j < n - 1; j++)
        {   
            filtered_image->data[i][j] = filte[i][j];
            //printf("%d ", filtered_image->data[i][j]);
        }
        //printf("\n");
    }
    
    // Free device memory
    cudaFree(pImage);
    cudaFree(pFilte);

    // Return the filter frame time
    return irun_time;
    
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
    if (stat("../filtered_cuda_gpu", &st) == -1)
    {
        mkdir("../filtered_cuda_gpu", 0700);
    }
    // Remove measurement files if they exist
    remove("../filtered_cuda_gpu/time.txt");
    remove("../filtered_cuda_gpu/memory.txt");
    remove("../filtered_cuda_gpu/gpu.txt");

    // File for runtime measurements
    FILE *fptr_time = fopen("../filtered_cuda_gpu/time.txt", "a");
    fprintf(fptr_time,"Frame Filtering Runtime Measurements\n\n");

    // File for memory usage measurements
    FILE *fptr_mem = fopen("../filtered_cuda_gpu/memory.txt", "a");
    fprintf(fptr_mem,"Frame Filtering GPU Memory Usage Measurements\n\n");

    // File for cpu usage measurements
    FILE *fptr_gpu = fopen("../filtered_cuda_gpu/gpu.txt", "a");
    fprintf(fptr_gpu,"Frame Filtering GPU Usage Measurements\n\n");

    // Obtain GPU memory initial use
    char *mem_command = "nvidia-smi --query-gpu=memory.used --format=csv | tail -1";
    char *mem_result = execute(mem_command);
    int mem_int = atoi(mem_result);
    
    // Set memory space for input frames
    Image *input_images = (Image*) malloc(batch_amount * sizeof(*input_images));
    
    // Set memory space for output frames
    Image *filtered_images = (Image*) malloc(batch_amount * sizeof(*filtered_images));

    // Variable to store full execution time
    float full_time = 0;

    // Travel the batches of frames
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

        // Filter the batch of frames
        for (int filter_c = 0; filter_c < batch_amount; filter_c++)
        {
            int file_number = filter_c + batches_c * batch_amount;
            // Call the filter function
            double frame_time = filter(&input_images[filter_c], &filtered_images[filter_c],
                                       1, file_number, fptr_time, fptr_mem, fptr_gpu, mem_int);
            // Adds frame time to full time
            full_time += frame_time;
            printf("Frame %d filtered.\n", file_number);
        }


        // Save and write the batch of frames
        for (int file_write_c = 0; file_write_c < batch_amount; file_write_c++)
        {
            int file_number=file_write_c+batches_c*batch_amount;
            char *filename;
            asprintf(&filename, "../filtered_cuda_gpu/frame%d.png", file_number);
            write_image(filename, &filtered_images[file_write_c]);
            printf("Frame %d saved.\n", file_number);
        }
    }

    // Write the full runtime of the process
    fprintf(fptr_time,"\nTotal Runtime = %f s", full_time/1000);
    //printf("Execution time: %f\n", full_time);

    // Close the files
    fclose(fptr_time);
    fclose(fptr_mem);
    fclose(fptr_gpu);

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