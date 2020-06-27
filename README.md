# Scalable CAIM discretization on multiple GPUs using concurrent kernels

Class-attribute interdependence maximization (CAIM) is one of the state-of-the-art algorithms for discretizing data for which classes are known. However, it may take a long time when run on high-dimensional large-scale data, with large number of attributes and/or instances. This paper presents a solution to this problem by introducing a graphic processing unit (GPU)-based implementation of the CAIM algorithm that significantly speeds up the discretization process on big complex data sets. The GPU-based implementation is scalable to multiple GPU devices and enables the use of concurrent kernels execution capabilities of modern GPUs. The CAIM GPU-based model is evaluated and compared with the original CAIM using single and multi-threaded parallel configurations on 40 data sets with different characteristics. The results show great speedup, up to 139 times faster using four GPUs, which makes discretization of big data efficient and manageable. For example, discretization time of one big data set is reduced from 2 h to < 2 min.

# Manuscript - Journal of Supercomputing

https://link.springer.com/article/10.1007/s11227-014-1151-8

# Citing CAIM-GPU

> A. Cano, S. Ventura, and K.J. Cios. Scalable CAIM discretization on multiple GPUs using concurrent kernels. Journal of Supercomputing, 69(1), 273-292, 2014.

## Running

Edit the Weka launcher to add the GPU library to the Java library path

```
-Djava.library.path=./src/main/resources/CAIMGPU
```