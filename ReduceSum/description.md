This project is a demo project for using GPU for global sum.

The array size is defined as 8M.

The project will use reduce sum and atomAdd for the implementation.

The following versions are implemented:
- The naive atomicAdd version
    - use only atomic add function
    - use reduce sum
    - use reduce sum with stream overlap