/*
Author: Vedanta Pawar
NetID: vp273
Class: M.Eng ECE, Cornell University
Email: vp273@cornell.edu

Instructions for Compiling and Executing Code:
Compile: /usr/local/cuda-10.1/bin/nvcc -o vp273_hw5_1.out vp273_hw5_1.cu
Run: ./vp273_hw5_1.out "Enter the dimension of the matrix:" "Enter the Block Size:"
Example: ./vp273_hw5_1.out 4096 8
*/

#include <stdio.h> 	// For printf() 
#include <stdint.h>	// For uint64
#include <stdlib.h> // For srand48() and drand48()
#include <time.h>   // For clock_gettime()

#define BILLION 1000000000L	  // To convert clock time in floating point seconds to nanoseconds/

__global__ void matrixMul( double** dev_matA , double** dev_matB , double** dev_matC , int ndim )
{
	double partial = 0.0;
	int i = blockIdx.y * blockDim.y + threadIdx.y; // Row i of C
	int j = blockIdx.x * blockDim.x + threadIdx.x; // Column j of C
	if ( i < ndim && j < ndim )
	{
		for ( int k = 0 ; k < ndim ; k++ )
		{	
			partial += dev_matA[i][k] * dev_matB[k][j];
		}
		dev_matC[i][j] = partial;
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
	double *matA[ndim] ;
	double *matB[ndim] ;
	double *matC[ndim] ;
	for ( int i = 0 ; i < ndim ; i ++ )
	{
		matA[i] = ( double * )malloc( ndim * sizeof( double ) );
		matB[i] = ( double * )malloc( ndim * sizeof( double ) );
		matC[i] = ( double * )malloc( ndim * sizeof( double ) );
	}

	clock_gettime( CLOCK_MONOTONIC , &start );
	// Iterate through the rows of the Matrix A and B
	for ( int i = 0 ; i < ndim ; i++ )
	{
		// Iterate through the columns of the Matrix A and B
		for ( int j = 0 ; j < ndim ; j++ )
		{
			clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
			diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
			srand48( diff ) ; // Set random seed to for initializing drand48() later
			// Store same random numbers in A and B
			matA[i][j] = drand48() ;
			matB[i][j] = drand48() ;
		}
	}

	double *dev_matA[ndim] , *dev_matB[ndim] , *dev_matC[ndim];
	for ( int i = 0 ; i < ndim ; i++ )
	{
		cudaMalloc( ( void** )&dev_matA[i], ndim * sizeof( double ) );
		cudaMalloc( ( void** )&dev_matB[i], ndim * sizeof( double ) );
		cudaMalloc( ( void** )&dev_matC[i], ndim * sizeof( double ) );
	}	

	// Start high resolution clock timer
	clock_gettime( CLOCK_MONOTONIC , &start );
	
	for ( int i = 0 ; i < ndim ; i++ )
	{
		cudaMemcpy( dev_matA[i] , matA[i] , ndim * sizeof( double ) , cudaMemcpyHostToDevice );
		cudaMemcpy( dev_matB[i] , matB[i] , ndim * sizeof( double ) , cudaMemcpyHostToDevice );
	}
	

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

	ptr_file = fopen("output_hw5_1.csv", "a");  // Save the time and corresponding matrix dim. in a csv file
	// Store the time and corresponding matrix dimension in a csv file
	fprintf( ptr_file ,"%d , %d , %llu\n", ndim , block_size ,  ( long long unsigned int ) diff );

	//Deallocate the memory allocated to matrices A, B and C
	for ( int i = 0 ; i < ndim ; i++ ) 
	{ 
		free ( matA[i] ) ;
		free ( matB[i] ) ;
		free ( matC[i] ) ;
		
	}
	cudaFree( dev_matA ) ;
	cudaFree( dev_matB ) ;
	cudaFree( dev_matC ) ;
	
	exit( 0 ) ;
}
