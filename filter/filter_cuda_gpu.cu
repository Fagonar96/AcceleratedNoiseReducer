#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include "handle_image_cuda.c"

#define WINDOW_SIZE 1
#define NEIGHBORHOOD_SIZE ((WINDOW_SIZE * 2 + 1) * (WINDOW_SIZE * 2 + 1))
struct stat st = {0};

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
    const int m = IMAGE_M;
    const int n = IMAGE_N;
    const int size = m * n * sizeof(int);

    memset(image, 0, size);
    memset(filte, 0, size);

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

    // Allocate matrix in device memory
    int (*pImage)[n], (*pFilte)[n];
    cudaMalloc((void**)&pImage, size);
    cudaMalloc((void**)&pFilte, size);

    // Copy matrices from host memory to device memory
    cudaMemcpy(pImage, image, size, cudaMemcpyHostToDevice);
    cudaMemcpy(pFilte, filte, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(m / threadsPerBlock.x, n / threadsPerBlock.y);
    //printf("numBlocks.x = %d, numBlocks.y = %d\n", numBlocks.x, numBlocks.y);

    float irun_time;
    cudaEvent_t istart, iend;
    cudaEventCreate(&istart);
    cudaEventCreate(&iend);
    cudaEventRecord(istart, 0);
    ImageFilter<<<numBlocks, threadsPerBlock>>>(pImage, pFilte, window_size);
    cudaEventRecord(iend, 0);
	cudaEventSynchronize(iend);
	cudaEventElapsedTime(&irun_time, istart, iend);
    printf("Frame time = %f\n", irun_time/1000);
    

    // Copy result from device memory to host memory
    cudaMemcpy(filte, pFilte, (m*n)*sizeof(int), cudaMemcpyDeviceToHost);
    
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
    if (stat("../filtered_cuda_gpu", &st) == -1)
    {
        mkdir("../filtered_cuda_gpu", 0700);
    }
    
    // Set memory space for input frames
    Image *input_images = (Image*) malloc(batch_amount * sizeof(*input_images));
    

    // Set memory space for output frames
    Image *filtered_images = (Image*) malloc(batch_amount * sizeof(*filtered_images));

    // Variable to store full execution time
    float full_time = 0;

    // Process the batch of frames
    int batches = file_amount/batch_amount;
    for (int batches_c = 0; batches_c < batches; batches_c++)
    {
        // Load and read the batch of frames
        for (int files_c = 0; files_c < batch_amount; files_c++)
        {
            int file_number=files_c+batches_c*batch_amount;
            char *filename;
            asprintf(&filename, "%s/frame%d.png", input_directory, file_number);
            printf("Frame with filename: %s read.\n", filename);
            Image *image = read_image( &input_images[files_c],filename);
        }

        float run_time;
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start, 0);

        // Filter the batch of frames
        for (int filter_c = 0; filter_c < batch_amount; filter_c++)
        {   
            // Call the filter function
            int file_number=filter_c+batches_c*batch_amount;
            filter(&input_images[filter_c], &filtered_images[filter_c], 1);
            printf("Frame %d filtered.\n", file_number);
        }

        cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&run_time, start, end);

        full_time += run_time;



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

    printf("Execution time: %f\n", full_time/1000);

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