/*
Author: Vedanta Pawar
NetID: vp273
Class: M.Eng ECE, Cornell University
Email: vp273@cornell.edu
Instructions for Compiling and Executing Code:
Compile: gcc vp273_hw2_code.c -std=gnu99 -o vp273_hw2_code
Run: ./vp273_hw2_code
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
	float p;
	int num;
	float x ;

	printf( "Enter the dimension of the matrix:\n" );
	scanf("%d" , &ndim); 	// Store matrix dimension in ndim 	

	/*
	Create matrix on heap by malloc and storing and assigning pointers to
	the matrix as matA.
	The matrix is a linear array of size ndim x ndim
	Total memory malloced is ndim^2
	 */
	double *matA = ( double * )malloc( ndim * ndim * sizeof( double ) ) ;
	double *matB = ( double * )malloc( ndim * sizeof( double ) ) ;
	 
	// Iterate through the rows of the Matrix A 
	for ( int i = 0 ; i < ndim ; i++ )
	{
		// Iterate through the columns of the Matrix A 
		for ( int j = 0 ; j < ndim ; j++ )
		{
			srand48( clock() ) ; // Set random seed to for initializing drand48() later
			// Store same random numbers in A 
			*( matA + i * ndim + j ) = drand48() ;		
			// scanf( "%d" , &num )	;
			// *( matA + i * ndim + j ) = num ;		
		}
		srand48 ( clock() ) ;
		*( matB + i ) = drand48() ;	
		// scanf( "%d" , &num )	;
		// *( matB + i ) = num ;	
	}
	
	// Start high resolution clock timer
	clock_gettime( CLOCK_MONOTONIC , &start );
	for ( int i = 0 ; i < ndim - 1 ; i++ )
	{
		for ( int j = i + 1 ; j < ndim ; j++ )
		{
			if ( * ( matA + i * ndim + i ) == 0 )
			{
				break ;
			}
			p = * ( matA + j * ndim + i ) / * ( matA + i * ndim + i ) ;
			for ( int k = i ; k < ndim ; k++ )
			{
				*( matA + j * ndim + k ) -= p * ( *( matA + i * ndim + k ) ) ;
			}
			*( matB + j ) -= p * ( *( matB + i ) ) ;
		}
	}

	for ( int i = ndim - 1 ; i >= 0 ; i-- )
	{
		if ( * (matA + i * ndim + i) == 0 )
		{
			printf( "No Solution Exists");
			break ;
		}
		x = *( matB + i ) / * (matA + i * ndim + i) ;
		// printf ( "Answer: %f \n" , x ) ; 
		for ( int k = 0 ; k <= i ; k++ )
		{
			*( matB + k ) -= x * ( *( matA + k * ndim + i ) ) ; 
		}
	} 

	// for ( int i = 0 ; i < ndim ; i++ )
	// {
	// 	// Iterate through the columns of the Matrix A 
	// 	for ( int j = 0 ; j < ndim ; j++ )
	// 	{
	// 		printf ( "%f \n" , *( matA + i * ndim + j ) ) ;
	// 	}
	// 	printf ( "%f \n" , *( matB + i ) ) ;
	// }

	clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
	//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	//Deallocate the memory allocated to matrices A, B and C
	free ( matA ) ;
	free ( matB ) ;
	exit( 0 ) ;
}
