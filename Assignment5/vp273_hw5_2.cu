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

__global__ void matrixMul( double* dev_matA , double* dev_matB , double* dev_matC , int ndim  )
{
	// extern __shared__ double A_tile[];
	// double *A_tile_local = ( double* ) A_tile ;
	// extern __shared__ double B_tile[];
	// double *B_tile_local = ( double* ) A_tile ;
	__shared__ double A_tile[16][16];
	__shared__ double B_tile[16][16];
	double partial = 0.0;
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;
	/* transpose B while loading to shared memory */
	for (int m = 0; m < ndim / blockDim.x ; ++m) 
	{
		A_tile[ ty ][ tx ] = dev_matA[ row * ndim + m*blockDim.x + tx ]; /* load coalesced */
		B_tile[ ty ][ tx ] = dev_matB[ ( m * blockDim.y + ty ) * ndim + col ]; /* not load coalesced */
		__syncthreads();
		for (int k = 0; k < blockDim.x; ++k)
		{
			partial += A_tile[ ty ][ k ] * B_tile[ k ][ tx ]; /*Bank conflicts */
		}
		__syncthreads();
		dev_matC[row * ndim + col] = partial; 
	}
}


void print( double* mat , int ndim )
{
	for( int i = 0 ; i < ndim ; i++ )
	{
		for ( int j = 0 ; j < ndim ; j++ )
		{
			printf( "%lf \t", mat[ i * ndim + j ] ) ;
		}
		printf( "\n" ) ;
	}
}

int main( int argc, char *argv[] )
{
	uint64_t diff; 	     		// Stores the time in nanoseconds
	struct timespec start, end; // Used to implement the high resolution timer included in <time.h>
	int ndim = atoi( argv[1] );					// Ask user and store the dimension of the square matrix
	int tile_size ;
	int number ;
	
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
			//Store same random numbers in A and B
			*( matA + i * ndim + j ) = drand48() ;
			*( matB + i * ndim + j ) = drand48() ;
			// scanf( "%d", &number ) ;
			// *( matA + i * ndim + j ) = number ;
			// *( matB + i * ndim + j ) = number ;
		}
	}

	cudaMalloc( ( void** )&dev_matA, ndim * ndim * sizeof( double ) );
	cudaMalloc( ( void** )&dev_matB, ndim * ndim * sizeof( double ) );
	cudaMalloc( ( void** )&dev_matC, ndim * ndim * sizeof( double ) );

	// Start high resolution clock timer
	clock_gettime( CLOCK_MONOTONIC , &start );
	
	cudaMemcpy( dev_matA , matA , ndim * ndim * sizeof( double ) , cudaMemcpyHostToDevice );
	cudaMemcpy( dev_matB , matB , ndim * ndim * sizeof( double ) , cudaMemcpyHostToDevice );

	
	tile_size = atoi( argv[2] ) ;
	dim3 Block(tile_size , tile_size) ;
	dim3 Grid( ndim / Block.x , ndim / Block.y) ;
	matrixMul<<< Grid, Block >>>( dev_matA , dev_matB , dev_matC , ndim );
	cudaMemcpy( matC , dev_matC , ndim * ndim * sizeof( double ) , cudaMemcpyDeviceToHost );	


	clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
	//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	// print( matC  , ndim ) ;
	//Deallocate the memory allocated to matrices A, B and C
	free ( matA ) ;
	free ( matB ) ;
	free ( matC ) ;
	cudaFree( dev_matA ) ;
	cudaFree( dev_matB ) ;
	cudaFree( dev_matC ) ;
	exit( 0 ) ;
}