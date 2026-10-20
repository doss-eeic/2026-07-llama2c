# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: run.c
	$(CC) -O3 -o run run.c -lm
	$(CC) -O3 -o runq runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: run.c
	$(CC) -g -o run run.c -lm
	$(CC) -g -o runq runq.c -lm

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(CC) -Ofast -o run run.c -lm
	$(CC) -Ofast -o runq runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -march=native runq.c  -lm  -o runq

# Basic OpenMP build (no profiling).
OMP_CFLAGS ?= -Ofast -fopenmp -march=native
.PHONY: ompbuild
ompbuild:
	mkdir -p build
	$(CC) $(OMP_CFLAGS) run.c -lm -o build/omp-run

# Separate detailed build; never overwrites build/omp-run.
DETAIL_CFLAGS ?= $(OMP_CFLAGS)
.PHONY: detailbuild
detailbuild:
	mkdir -p build
	$(CC) $(DETAIL_CFLAGS) -DLLAMA_PROFILE run_detail.c -lm -o build/omp-run-detail

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq

# Persistent CPU-pinned worker pool (Linux); separate from reference/detail builds.
STAY_CFLAGS ?= -Ofast -march=native -pthread
.PHONY: staybuild stayprofile
staybuild: run_stay.c run_detail.c
	mkdir -p build
	$(CC) $(STAY_CFLAGS) run_stay.c -lm -o build/stay-run

stayprofile: run_stay.c run_detail.c
	mkdir -p build
	$(CC) $(STAY_CFLAGS) -DLLAMA_PROFILE run_stay.c -lm -o build/stay-run-detail

# Fixed layer team with output-only CPU switching.
.PHONY: stay2build stay2profile
stay2build: run_stay2.c run_detail.c
	mkdir -p build
	$(CC) $(STAY_CFLAGS) run_stay2.c -lm -o build/stay2-run

stay2profile: run_stay2.c run_detail.c
	mkdir -p build
	$(CC) $(STAY_CFLAGS) -DLLAMA_PROFILE run_stay2.c -lm -o build/stay2-run-detail
