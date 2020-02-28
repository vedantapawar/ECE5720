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


void find_max ( double **matA , int i , int ndim , double *matB) 
{
	int i_max = i + 1 ; // For storing max i
	double max = matA [ i + 1 ][ i ] ;
	for ( int j = i + 1 ; j < ndim ; j++ )
	{
		if ( max < matA[ j ][ i ] )
		{
			max = matA[ j ][ i ] ;
			i_max = j ;	
		}
	}
	double *temp ;
	temp = matA [ i + 1 ] ;
	matA [ i + 1 ] = matA [ i_max ] ;
	matA [ i_max ] = temp ;
	double temp_num;
	temp_num = *(matB + i + 1) ;
	*(matB + i + 1) = *(matB + i_max) ;
	*(matB + i_max) = temp_num ;
}

void print ( double **matA , int ndim )
{
	for(int i = 0 ; i < ndim ; 	i++)
	{
		for(int j = 0 ; j < ndim ; j++)
		{
			printf("%lf\n" , matA[ i ][ j ]);
		}
	}
}
int main( int argc, char *argv[] )
{
	uint64_t diff; 				// Stores the time in nanoseconds
	struct timespec start, end; // Used to implement the high resolution timer included in <time.h>
	int ndim ;					// Ask user and store the dimension of the square matrix
	double sum = 0.0;			// Intermediate sum of the innerproduct of the row vector and column vector
	int num ;
	double p;
	double rho = 0.0 ;
	// printf( "Enter the dimension of the matrix:\n" );
	// scanf("%d" , &ndim); 	// Store matrix dimension in ndim 	
	ndim = atoi ( argv[ 1 ] ) ;

	/*
	Create matrix on heap by malloc and storing and assigning pointers to
	the matrix as matA.
	The matrix is a linear array of size ndim x ndim
	Total memory malloced is ndim^2
	 */

	double *matA [ ndim ] ;
	double *matA_check [ ndim ] ;
	for ( int i = 0 ; i < ndim ; i++ )
	{
		matA [ i ] = (double *)malloc( ndim * sizeof(double) );
		matA_check [ i ] = (double *)malloc( ndim * sizeof(double) );
	}
	double *matB = ( double * )malloc( ndim * sizeof( double ) ) ;
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
			matA[ i ][ j ] = drand48() ;		
			// scanf( "%d" , &num )	;
			// matA[ i ][ j ] = num ;	
			matA_check[ i ][ j ] = matA[ i ][ j ] ;
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
		// printf("Iteration\n");
		find_max ( matA , i , ndim , matB) ;
		// print ( matA , ndim ) ;
		for ( int j = i + 1 ; j < ndim ; j++ )
		{
			if ( matA[ i ][ i ] == 0 )
			{
				break ;
			}
			p = matA[ j ][ i ] / matA[ i ][ i ] ;
			for ( int k = i ; k < ndim ; k++ )
			{
				matA[ j ][ k ] -= p * ( matA[ i ][ k ] ) ;
			}
			*( matB + j ) -= p * ( *( matB + i ) ) ;
			// printf("p %lf \n" , p);
		}
	}
	// printf("Triangle\n");
	// print(matA , ndim) ;
	for ( int i = ndim - 1 ; i >= 0 ; i-- )
	{
		// if ( * (matA + i * ndim + i) == 0 )
		// {
		// 	printf( "No Solution Exists");
		// 	break ;
		// }
		*( x + i ) = *( matB + i ) / matA[ i ][ i ] ;
		// printf ( "Answer: %f \n" , *( x + i ) ) ; 
		for ( int k = 0 ; k <= i ; k++ )
		{
			*( matB + k ) -= *( x + i ) * ( matA[ k ][ i ] ) ; 
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
			sum += matA_check[ i ][ j ] * ( *( x + j ) ) ;
			// printf("%lf , %lf \n" , *( matA_check + i * ndim + j ) , ( *( x + j )) );
		}
		// printf("%lf 	, %lf \n" , sum , *(matB_check + i ) );
		rho += pow ( ( sum - *(matB_check + i ) ) , 2 )  ;
	}

	rho = sqrt ( rho ) ;
	printf ( "The error is %lf\n" , rho ) ;


	//Deallocate the memory allocated to matrices A, B and C
	for ( int i = 0 ; i < ndim ; i++ ) 
	{ 
		free(matA[ i ]);
		free(matA_check[ i ]);
	}
	// free ( matA ) ;
	free ( matB ) ;
	// free ( matA_check ) ;
	free ( matB_check ) ;
	free ( x ) ;
	exit( 0 ) ;
}
