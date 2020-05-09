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

int main( int argc, char *argv[] )
{
	uint64_t diff; 				// Stores the time in nanoseconds
	struct timespec start, end; // Used to implement the high resolution timer included in <time.h>
	int ndim ;					// Ask user and store the dimension of the square matrix
	float sum = 0.0;			// Intermediate sum of the innerproduct of the row vector and column vector

	printf( "Enter the dimension of the matrix to be multiplied:\n" );
	scanf("%d" , &ndim); 	// Store matrix dimension in ndim 	

	/*
	Create matrices on heap by malloc and storing and assigning pointers to
	the individual matrices as matA, matB, matC.
	Each matrix is a linear array of size ndim x ndim
	Total memory malloced is 3 x ndim^2
	 */
	double *matA = ( double * )malloc( ndim * ndim * sizeof( double ) );
	double *matB = ( double * )malloc( ndim * ndim * sizeof( double ) );
	double *matC = ( double * )malloc( ndim * ndim * sizeof( double ) );

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

    // Start high resolution clock timer
	clock_gettime( CLOCK_MONOTONIC , &start );

	//Iterate through the rows of A and C
	for ( int i = 0 ; i < ndim ; i++ )
	{
		// Iterate through the columns B and C
		for ( int j = 0; j < ndim ; j++ )
		{
			// Iterate through the columns of A and Rows of B
			for ( int k = 0 ; k < ndim ; k++)
			{
				// Multiply the two elements
				*( matC + i * ndim + j ) += *( matA + i * ndim + k ) * ( *( matB + k * ndim + j ) ) ;
			}
		}
	}

	clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
	//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	//Deallocate the memory allocated to matrices A, B and C
	free ( matA ) ;
	free ( matB ) ;
	free ( matC ) ;
	exit( 0 ) ;
}