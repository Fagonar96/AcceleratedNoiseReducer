LDLIBS = -lpng -lm -fopenmp -w

PROGRAMS = filter filter-omp-cpu filter-omp-gpu filter-oacc-cpu filter-oacc-gpu filter-cuda-gpu

all: $(PROGRAMS)

configure: configure.c
	gcc configure.c -o configure
	./configure

filter: filter.c
	nvc filter.c $(LDLIBS) -o filter
	./filter ../video 16 4

filter-omp-cpu: filter_omp_cpu.c
	nvc -mp=multicore filter_omp_cpu.c $(LDLIBS) -o filter-omp-cpu
	./filter-omp-cpu ../video 16 4

filter-omp-gpu: filter_omp_gpu.c
	nvc -mp=gpu filter_omp_gpu.c $(LDLIBS) -o filter-omp-gpu
	./filter-omp-gpu ../video 16 4

filter-oacc-cpu: filter_oacc_cpu.c 
	nvc -acc=multicore filter_oacc_cpu.c $(LDLIBS) -o filter-oacc-cpu
	./filter-oacc-cpu ../video 16 4

filter-oacc-gpu: filter_oacc_gpu.c
	nvc -acc filter_oacc_gpu.c $(LDLIBS) -o filter-oacc-gpu
	./filter-oacc-gpu ../video 16 4

filter-cuda-gpu: filter_cuda_gpu.cu
	nvcc filter_cuda_gpu.cu -lpng -lm -w -o filter-cuda-gpu
	cuda-memcheck ./filter-cuda-gpu ../video 16 4
clean:
	$(RM) *.o $(PROGRAMS) &> /dev/null || true
