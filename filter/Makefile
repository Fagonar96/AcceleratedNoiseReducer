LDLIBS = -lpng -lm -fopenmp -w

PROGRAMS = filter filter-omp-cpu filter-omp-fpga filter-oacc-cpu filter-oacc-fpga

all: $(PROGRAMS)

filter: filter.c
	nvc filter.c $(LDLIBS) -o filter
	./filter ../video 16 4

filter-omp-cpu: filter_omp_cpu.c
	nvc -mp=multicore filter_omp_cpu.c $(LDLIBS) -o filter-omp-cpu
	./filter-omp-cpu ../video 16 4

filter-omp-fpga: filter_omp_cpu.c
	gcc filter_omp_cpu.c $(LDLIBS) -o filter-omp-fpga
	./filter-omp-fpga ../video 16 4

filter-oacc-cpu: filter_oacc_cpu.c 
	nvc -acc=multicore filter_oacc_cpu.c $(LDLIBS) -o filter-oacc-cpu
	./filter-oacc-cpu ../video 16 4

filter-oacc-fpga: filter_oacc_cpu.c
	gcc filter_oacc_cpu.c -lpng -lm -w -fopenacc -foffload=disable -o filter-oacc-fpga
	./filter-oacc-fpga ../video 16 4



clean:
	$(RM) *.o $(PROGRAMS) &> /dev/null || true
