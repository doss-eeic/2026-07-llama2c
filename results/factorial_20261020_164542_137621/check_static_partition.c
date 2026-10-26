#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
int main(void) {
    int owners[864];
    const int counts[] = {1,4,8,10,14}, sizes[] = {6,24,31,37,864};
    for(int c=0;c<5;c++) for(int s=0;s<5;s++) {
        int n=counts[c],rows=sizes[s];
        #pragma omp parallel for num_threads(n) schedule(static)
        for(int i=0;i<rows;i++) owners[i]=omp_get_thread_num();
        for(int rank=0;rank<n;rank++) {
            int q=rows/n,r=rows%n,begin=rank*q+(rank<r?rank:r),end=begin+q+(rank<r);
            for(int i=begin;i<end;i++) if(owners[i]!=rank)return 1;
        }
    }
    puts("GCC OpenMP static partition matches experimental pthread partition");
}
