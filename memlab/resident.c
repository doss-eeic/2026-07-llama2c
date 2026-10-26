// prints the fraction of each file that is in the page cache (like util-linux fincore)
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
int main(int argc, char** argv) {
    for (int a = 1; a < argc; a++) {
        int fd = open(argv[a], O_RDONLY); struct stat st; fstat(fd, &st);
        size_t n = (st.st_size + 4095) / 4096;
        void* p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        unsigned char* v = malloc(n); mincore(p, st.st_size, v);
        size_t r = 0; for (size_t i = 0; i < n; i++) r += v[i] & 1;
        printf("%5.1f%%  %s\n", 100.0 * r / n, argv[a]);
    }
    return 0;
}
