#include <omp.h>
#include <stdio.h>
int main(void){omp_sched_t s;int c;omp_get_schedule(&s,&c);printf("{\"max_threads\":%d,\"schedule_kind\":%u,\"chunk\":%d}\n",omp_get_max_threads(),(unsigned)s,c);}
