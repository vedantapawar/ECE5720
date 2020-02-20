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
#include <math.h>

#define BILLION 1000000000L	  // To convert clock time in floating point seconds to nanoseconds/

/*
void find_max ( double *matA , int i , int ndim ) 
{
	int i_max = i + 1 ; // For storing max i
	double max = *( matA + i * ndim + i ) ;
	for ( int j = i + 1 ; j < ndim ; j++ )
	{
		if ( max < *( matA + j * ndim + i ) )
		{
			max = *( matA + j * ndim + i ) ;
			i_max = j ;
		}
	}
	double *temp ;
	temp = matA + i * ndim ;
	matA + i * ndim = matA + i_max ;

}*/

int main( int argc, char *argv[] )
{
	uint64_t diff; 				// Stores the time in nanoseconds
	struct timespec start, end; // Used to implement the high resolution timer included in <time.h>
	int ndim ;					// Ask user and store the dimension of the square matrix
	double sum = 0.0;			// Intermediate sum of the innerproduct of the row vector and column vector
	int num ;
	double p;
	double rho = 0.0 ;
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
	double *matA_check = ( double * )malloc( ndim * ndim * sizeof( double ) ) ;
	double *matB_check = ( double * )malloc( ndim * sizeof( double ) ) ;
	double* x = ( double * )malloc( ndim * sizeof( double ) ) ;

	// Iterate through the rows of the Matrix A 
	clock_gettime( CLOCK_MONOTONIC , &start );
	for ( int i = 0 ; i < ndim ; i++ )
	{
		// Iterate through the columns of the Matrix A 
		for ( int j = 0 ; j < ndim ; j++ )
		{
			clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
			//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
			diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
			srand48( diff ) ; // Set random seed to for initializing drand48() later
			// Store same random numbers in A 
			*( matA + i * ndim + j ) = drand48() ;		
			// scanf( "%d" , &num )	;
			// *( matA + i * ndim + j ) = num ;	
			*( matA_check + i * ndim + j ) = *( matA + i * ndim + j ) ;
		}
			clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
			//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
			diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
			srand48( diff ) ; // Set random seed to for initializing drand48() later		*( matB + i ) = drand48() ;	
		// scanf( "%d" , &num )	;
		// *( matB + i ) = num ;
		*( matB + i ) = drand48() ;	
		* ( matB_check + i ) = *( matB + i ) ;
	}
	
	// Start high resolution clock timer
	clock_gettime( CLOCK_MONOTONIC , &start );
	for ( int i = 0 ; i < ndim - 1 ; i++ )
	{
		// find_max ( matA , i , ndim ) ;
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
			// printf("p %lf \n" , p);
		}
	}

	for ( int i = ndim - 1 ; i >= 0 ; i-- )
	{
		// if ( * (matA + i * ndim + i) == 0 )
		// {
		// 	printf( "No Solution Exists");
		// 	break ;
		// }
		*( x + i ) = *( matB + i ) / * (matA + i * ndim + i) ;
		printf ( "Answer: %f \n" , *( x + i ) ) ; 
		for ( int k = 0 ; k <= i ; k++ )
		{
			*( matB + k ) -= *( x + i ) * ( *( matA + k * ndim + i ) ) ; 
		}
	} 

	clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
	//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	// for (int i = 0; i < ndim; i++)
	// {
	// 	printf("%lf\n", *(x+i));
	// }
	
	printf ( "___________Numerical Verification___________\n" ) ;
	for ( int i = 0 ; i < ndim ; i++ )
	{
		sum = 0.0 ;
		for ( int j = 0 ; j < ndim ; j++ )
		{
			sum += *( matA_check + i * ndim + j ) * ( *( x + j ) ) ;
			// printf("%lf , %lf \n" , *( matA_check + i * ndim + j ) , ( *( x + j )) );
		}
		// printf("%lf 	, %lf \n" , sum , *(matB_check + i ) );
		rho += pow ( ( sum - *(matB_check + i ) ) , 2 )  ;
	}

	rho = sqrt ( rho ) ;
	printf ( "The error is %lf\n" , rho ) ;


	//Deallocate the memory allocated to matrices A, B and C
	free ( matA ) ;
	free ( matB ) ;
	free ( matA_check ) ;
	free ( matB_check ) ;
	free ( x ) ;
	exit( 0 ) ;
}
