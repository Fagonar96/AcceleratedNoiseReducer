CC = gcc
LDLIBS = -lm -fopenmp -w

PROGRAMS = filter filter-omp-fpga

all: $(PROGRAMS)

filter: filter.c
	gcc -std=c99 filter.c $(LDLIBS) -o filter
	./filter video 32 2

filter-omp-fpga: filter_omp_fpga.c
	gcc -std=c99 filter_omp_fpga.c $(LDLIBS) -o filter-omp-fpga
	./filter-omp-fpga video 32 2

clean:
	$(RM) *.o $(PROGRAMS) &> /dev/null || true