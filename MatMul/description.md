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
- Half version
  - Based on the naive version, switch the data type from float to half
- SIMT verison
  - what is `__align__(16*1024)` used for?
  - why we need to add some buffer into the share memory?
  - c++ inline assumble and force inline?
- SIMT - pipeline version
- Tensorcore version
  - with pipeline
  - reference: https://github.com/jin-yc10/sparse_gemm/tree/master @jin-yc10
- Tensorcore with selected input output indexes (spconv)
- baseline CUTLASS