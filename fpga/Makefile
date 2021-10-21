CC = gcc
LDLIBS = -lm -fopenmp -w

PROGRAMS = filter-omp-fpga

all: $(PROGRAMS)

filter-omp-fpga: filter_omp_fpga.c
	gcc -std=c99 filter_omp_fpga.c $(LDLIBS) -o filter-omp-fpga
	./filter-omp-fpga video 1 1

clean:
	$(RM) *.o $(PROGRAMS) &> /dev/null || true