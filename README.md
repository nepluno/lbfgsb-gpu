Nonlinear optimization is at the heart of many algorithms in engineering. Recently, due to the rise of general purpose graphics processing unit (GPGPU), it is promising to investigate the performance improvement of optimization methods after parallelized. While much has been done for simple optimization methods such as conjugate gradient, due to the strong dependencies contained, little has been done for other more sophisticated ones, such as the limited memory Broyden-Fletcher-Goldfarb-Shanno with boundaries (L-BFGS-B). In this software, for the first time, a parallelized implementation of L-BFGS-B on the GPU is introduced. We show how to remove those dependencies, and also demonstrate its significant speed-up for practical applications.

Note:

Recently updated to use CUDA 5.5

Currently, there are two versions in the repository:

To obtain the version specifically optimized for computing the centroidal Voronoi tessellation (CVT), please use the code under folder "voronoi"

To obtain the version for solving general problems, please use the code under folder "general"

Related Works and Acknowledgement:

The code for rendering the Voronoi diagram is credited to Guodong Rong (https://sites.google.com/site/rongguodong/).

The original L-BFGS-B code on the CPU is credited to Argonne National Laboratory and Northwestern University. Their website is http://users.eecs.northwestern.edu/~nocedal/lbfgsb.html

Thanks them for sharing the code!

How to Configure the Environment for the Example program:

For just L-BFGS-B:

1. You'd have a video card whose compute capability is higher than 1.3.

2. Install the CUDA Toolkit v5.5 (if you're using another version, you'll have to re-customize the building process.) from the NVIDIA developer site.

3. Copy the L-BFGS-B folder and add all the files into your project, and set in your main cpp for some external variables in the lbfgsb.h.

4. Correct all the path of include files and libraries in the properties page of the project.

5. Enjoy it!

To compile the CVT-GPU example:

6. You also have to install the NVIDIA Cg Toolkit from the NVIDIA developer site.

PS: the current version supports Hessian approximation m <= 8, which has been proved to be sufficient for most applications.