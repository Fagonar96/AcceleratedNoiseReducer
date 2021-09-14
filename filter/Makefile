LDLIBS = -lpng -lm -fopenmp

PROGRAMS = filter filter-omp-cpu

all: $(PROGRAMS)

filter: filter.c
	nvc filter.c $(LDLIBS) -o filter
	./filter ../video 16 4


clean:
	$(RM) *.o $(PROGRAMS) &> /dev/null || true
