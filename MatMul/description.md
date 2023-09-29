This project is a demo project for using GPU for matrix multiplication.

The matrix is defined as 2048 * 512

We need to take the followings into consideration:
- Pipelineing
- Better memory access pattern
  - Share memory
  - Inplicit matrix transform
  - Memory coherence
  - Bank conflict
- Tensorcore
- ...

The following versions are implemented:
- Naive version
  - Using the naive for-loop algorithm
  - With a small optimization that gain 2.28x speed up
- SIMT verison
- SIMT - pipeline version
- Tensorcore - pipeline version