#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void start_timer(event_pair * p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}

inline float stop_timer(event_pair * p)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}