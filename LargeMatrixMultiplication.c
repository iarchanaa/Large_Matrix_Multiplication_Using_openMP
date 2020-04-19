#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>

typedef double TYPE;
#define MAX_DIM 2000*2000
#define MAX_VAL 10
#define MIN_VAL 1

/*Method to generate random square matrix for matrix computation
 assigning MIN_VAL 1 to MAX_VAL 10 */
TYPE** randomSquareMatrix(int dimension);
//Method to store the result of matrix multiplication 
TYPE** zeroSquareMatrix(int dimension);

// Matrix multiplication methods
double ParallelWithoutPragma(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);
double ParallelWithPragma(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int dimension);

void displaySquareMatrix(TYPE** matrix, int dimension);
void convert(TYPE** matrixA, TYPE** matrixB, int dimension);

// Specifies Dimension of Matrix
void DimensionOfMatrix(int dimension);

// 1 Dimensional matrix on stack
TYPE flatA[MAX_DIM];
TYPE flatB[MAX_DIM];

int main(int argc, char* argv[]){
	/*
	Creating 2D-Matrix of size 200 to 2000 increasing with size 200
	*/
	for(int dimension=200; dimension<=2000; dimension+=200){
		DimensionOfMatrix(dimension);
	}

	return 0;
}

TYPE** randomSquareMatrix(int dimension){
	/*
		Generate 2 dimensional random TYPE matrix.
	*/

	TYPE** matrix = malloc(dimension * sizeof(TYPE*));

	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}

	#pragma omp parallel for
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = rand() % MAX_VAL + MIN_VAL;
		}
	}

	return matrix;
}


TYPE** zeroSquareMatrix(int dimension){
	/*
		Generate 2 dimensional zero TYPE matrix.

	*/

	TYPE** matrix = malloc(dimension * sizeof(TYPE*));

	for(int i=0; i<dimension; i++){
		matrix[i] = malloc(dimension * sizeof(TYPE));
	}

	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			matrix[i][j] = 0;
		}
	}

	return matrix;
}


double ParallelWithPragma(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int size){
	/*
		Parallel multiply given input matrices and return resultant matrix
	*/
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
	float tmp;
	int chunk = 2;
	int tid;
	int block_size = 16;
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
#pragma omp parallel shared(matrixA, matrixB, matrixC, size, chunk) private(i, j, k, jj, kk, tid, tmp)
	{
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			printf ("Number of threads with pragma             |      %d",omp_get_num_threads());
			printf ("\n");
		}
		#pragma omp for schedule (static, chunk)
		for (jj = 0; jj < size; jj += block_size)
		{
			
			for (kk = 0; kk < size; kk += block_size)
			{
				for (i = 0; i < size; i++)
				{
					for (j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
					{
						tmp = 0.0f;
						for (k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
						{
							tmp += matrixA[i][k] * matrixB[k][j];
						}
						matrixC[i][j] += tmp;
					}
				}
			}
		}
	}
	/* Tail */

	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}


double ParallelWithoutPragma(TYPE** matrixA, TYPE** matrixB, TYPE** matrixC, int size){
	/*
		Parallel multiply given input matrices and return resultant matrix
	*/
	int i = 0, j = 0, k = 0, jj = 0, kk = 0;
	float tmp;
	int tid;
	int block_size = 16;
	struct timeval t0, t1;
	gettimeofday(&t0, 0);
	tid = omp_get_thread_num();
		
	{
		tid = omp_get_thread_num();
		if (tid == 0)
		{
			printf ("Number of thread without pragma           |      %d",omp_get_num_threads());
			printf ("\n");
		}
		for (jj = 0; jj < size; jj += block_size)
		{
			
			for (kk = 0; kk < size; kk += block_size)
			{
				for (i = 0; i < size; i++)
				{
					for (j = jj; j < ((jj + block_size) > size ? size : (jj + block_size)); j++)
					{
						tmp = 0.0f;
						for (k = kk; k < ((kk + block_size) > size ? size : (kk + block_size)); k++)
						{
							tmp += matrixA[i][k] * matrixB[k][j];
						}
						matrixC[i][j] += tmp;
					}
				}
			}
		}
	}
	/* Tail */

	gettimeofday(&t1, 0);
	double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;

	return elapsed;
}

void convert(TYPE** matrixA, TYPE** matrixB, int dimension){
	#pragma omp parallel for
	for(int i=0; i<dimension; i++){
		for(int j=0; j<dimension; j++){
			flatA[i * dimension + j] = matrixA[i][j];
			flatB[j * dimension + i] = matrixB[i][j];
		}
	}
}
void DimensionOfMatrix(int dimension){

	// Console write
	
	printf("___________________________________________________________\n");
	
	printf("Dimension | %d x %d\n", dimension,dimension);
	printf("__________|");
	printf("________________________________________________\n");

	double* opmLatency = malloc(4 * sizeof(double));
	TYPE** matrixA = randomSquareMatrix(dimension);
	TYPE** matrixB = randomSquareMatrix(dimension);
	
	// Iterate and measure performance
		TYPE** matrixResult = zeroSquareMatrix(dimension);
		opmLatency[0] = ParallelWithoutPragma(matrixA, matrixB, matrixResult, dimension);
		opmLatency[1] = ParallelWithPragma(matrixA, matrixB, matrixResult, dimension);
		free(matrixResult);

		// Console write
		printf("Matrix Multiplication with pragma         |\t%f\n", opmLatency[1]);

		printf("Matrix multiplication without pragma      |\t%f\n", opmLatency[0]);
		
	free(opmLatency);
	free(matrixA);
	free(matrixB);
}