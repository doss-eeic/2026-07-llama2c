# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
CPPFLAGS ?= -I.
RUN_SRC = run/run.c
RUN_HEADERS = src/fused.h src/classifier.h src/pool.h

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: $(RUN_SRC) $(RUN_HEADERS) | build
	$(CC) $(CPPFLAGS) -O3 -o build/run-fused $(RUN_SRC) -lm
	$(CC) $(CPPFLAGS) -O3 -o runq runq.c -lm

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./build/run-fused out/model.bin -n 3
rundebug: $(RUN_SRC) $(RUN_HEADERS) | build
	$(CC) $(CPPFLAGS) -g -o build/run-fused $(RUN_SRC) -lm
	$(CC) $(CPPFLAGS) -g -o runq runq.c -lm

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
runfast: $(RUN_SRC) $(RUN_HEADERS) | build
	$(CC) $(CPPFLAGS) -Ofast -o build/run-fused $(RUN_SRC) -lm
	$(CC) $(CPPFLAGS) -Ofast -o runq runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./build/run-fused out/model.bin
.PHONY: runomp
runomp: $(RUN_SRC) $(RUN_HEADERS) | build
	$(CC) $(CPPFLAGS) -Ofast -fopenmp -march=native $(RUN_SRC)  -lm  -o build/run-fused
	$(CC) $(CPPFLAGS) -Ofast -fopenmp -march=native runq.c  -lm  -o runq

.PHONY: win64
win64: $(RUN_SRC) $(RUN_HEADERS) | build
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o build/run-fused.exe -I. $(RUN_SRC) win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu: $(RUN_SRC) $(RUN_HEADERS) | build
	$(CC) $(CPPFLAGS) -Ofast -std=gnu11 -o build/run-fused $(RUN_SRC) -lm
	$(CC) $(CPPFLAGS) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu: $(RUN_SRC) $(RUN_HEADERS) | build
	$(CC) $(CPPFLAGS) -Ofast -fopenmp -std=gnu11 $(RUN_SRC)  -lm  -o build/run-fused
	$(CC) $(CPPFLAGS) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for $(RUN_SRC) C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) $(CPPFLAGS) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f build/run-fused build/run-upstream build/run-fused.exe
	rm -f runq

# Unmodified upstream baseline, built with the same OpenMP optimization flags.
.PHONY: runupstream
runupstream: run/run_upstream.c | build
	$(CC) $(CPPFLAGS) -Ofast -fopenmp -march=native $< -lm -o build/run-upstream

build:
	mkdir -p $@
