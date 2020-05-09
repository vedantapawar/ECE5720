/*
Author: Vedanta Pawar
NetID: vp273
Class: M.Eng ECE, Cornell University
Email: vp273@cornell.edu

Instructions for Compiling and Executing Code:
Compile: gcc vp273_mm_rbyc.c -std=gnu99 -o vp273_mm_rbyc
Run: ./vp273_mm_rbyc
*/

#include <stdio.h> 	// For printf() 
#include <stdint.h>	// For uint64
#include <stdlib.h> // For srand48() and drand48()
#include <time.h>   // For clock_gettime()

#define BILLION 1000000000L	  // To convert clock time in floating point seconds to nanoseconds/

__global__ void matrixMul( double* dev_matA , double* dev_matB , double* dev_matC , int ndim )
{
	double partial = 0.0;
	int i = blockIdx.y * blockDim.y + threadIdx.y; // Row i of C
	int j = blockIdx.x * blockDim.x + threadIdx.x; // Column j of C
	if ( i < ndim && j < ndim )
	{
		for ( int k = 0 ; k < ndim ; k++ )
		{	
			partial += dev_matA[i * ndim + k] * dev_matB[k * ndim + j];
		}
		dev_matC[i * ndim + j] = partial;
	}
	
}

int main( int argc, char *argv[] )
{
	uint64_t diff; 				// Stores the time in nanoseconds
	struct timespec start, end; // Used to implement the high resolution timer included in <time.h>
	int ndim = atoi( argv[1] );	// Ask user and store the dimension of the square matrix
	FILE *ptr_file ;            // File pointer used to point to a .csv file for storing the time vs ndim
	int block_size ;
	/*
	Create matrices on heap by malloc and storing and assigning pointers to
	the individual matrices as matA, matB, matC.
	Each matrix is a linear array of size ndim x ndim
	Total memory malloced is 3 x ndim^2
	 */
	double *matA = ( double * )malloc( ndim * ndim * sizeof( double ) );
	double *matB = ( double * )malloc( ndim * ndim * sizeof( double ) );
	double *matC = ( double * )malloc( ndim * ndim * sizeof( double ) );
	double *dev_matA , *dev_matB , *dev_matC;

	srand48 (1); 	// Set random seed to for initializing drand48() later

	// Iterate through the rows of the Matrix A and B
	for ( int i = 0 ; i < ndim ; i++ )
	{
		// Iterate through the columns of the Matrix A and B
		for ( int j = 0 ; j < ndim ; j++ )
		{
			// Store same random numbers in A and B
			*( matA + i * ndim + j ) = drand48() ;
			*( matB + i * ndim + j ) = drand48() ;
		}
	}

	cudaMalloc( ( void** )&dev_matA, ndim * ndim * sizeof( double ) );
	cudaMalloc( ( void** )&dev_matB, ndim * ndim * sizeof( double ) );
	cudaMalloc( ( void** )&dev_matC, ndim * ndim * sizeof( double ) );

	// Start high resolution clock timer
	clock_gettime( CLOCK_MONOTONIC , &start );

	cudaMemcpy( dev_matA , matA , ndim * ndim * sizeof( double ) , cudaMemcpyHostToDevice );
	cudaMemcpy( dev_matB , matB , ndim * ndim * sizeof( double ) , cudaMemcpyHostToDevice );

	block_size = atoi( argv[2] ) ;
	dim3 Block( block_size , block_size) ;
	dim3 Grid( ndim / Block.x , ndim / Block.y) ;
	matrixMul<<< Grid, Block>>>( dev_matA , dev_matB , dev_matC , ndim );
	cudaMemcpy( matC , dev_matC , ndim * ndim * sizeof( double ) , cudaMemcpyDeviceToHost );	

	cudaDeviceSynchronize( );

	clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
	//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	ptr_file = fopen("output_hw5_1.csv", "a");  // Save the time and corresponding stride in a csv file
	// Store the time and corresponding matrix dimension in a csv file
	fprintf( ptr_file ,"%d , %llu\n", ndim ,   ( long long unsigned int ) diff );

	//Deallocate the memory allocated to matrices A, B and C
	free ( matA ) ;
	free ( matB ) ;
	free ( matC ) ;
	cudaFree( dev_matA ) ;
	cudaFree( dev_matB ) ;
	cudaFree( dev_matC ) ;
	exit( 0 ) ;
}