# Large_Matrix_Multiplication_Using_openMP
PROBLEM STATEMENT: To develop an efficient large matrix multiplication algorithm in OpenMP.   LARGE MATRIX MULTIPLICATION: The goal of this assignment is to obtain the multiplication of a large two-dimension Matrix (2-D Matrix). There are several ways for computing the matrix multiplication but a blocked approach which is also called the partition approach seems to be a better solution than the traditional sequential approach. Here basically the idea is to break the complex arrays into something easier to deal with. The matrix is divided into multiple partitions, each thread performs the multiplication of the elements, reside in that simpler partition and finally sums up the results in a parallel fashion. 
