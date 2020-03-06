/*
Author: Vedanta Pawar
NetID: vp273
Class: M.Eng ECE, Cornell University
Email: vp273@cornell.edu

Instructions for Compiling and Executing Code:
Compile: gcc vp273_hw3_code.c -o vp273_hw3_code -std=gnu99 -fopenmp -lm
Run: ./vp273_hw3_code "Enter the dimension of the matrix:" "Enter the number of threads:"
Example: ./vp273_hw3_code 4096 8
*/

#include <stdio.h> 	 /* For printf()  */
#include <stdint.h>	 /* For uint64 */
#include <stdlib.h>  /* For srand48() and drand48() */
#include <time.h>    /* For clock_gettime() */
#include <omp.h> /* For OpenMP */
#include <math.h> 	 /* For verification of results */

#define BILLION 1000000000L	  /* To convert clock time in floating point seconds to nanoseconds */
int NUM_THREADS ;	  		  /* No. threads will be created */

void find_max ( double **matA , int i , int ndim , double **I) 
{
	/*Used to find the max of the row and swap with the current row*/
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
	temp = I [ i + 1 ] ;
	I [ i + 1 ] = I [ i_max ] ;
	I [ i_max ] = temp ;
}

int main( int argc, char *argv[] )
{
	uint64_t diff; 				/* Stores the time in nanoseconds */
	struct timespec start, end; /* Used to implement the high resolution timer included in <time.h> */
	int ndim ;					/* Ask user and store the dimension of the square matrix */
	int chunk ;
	int j = 0 , k , l;
	/* "Enter the dimension of the matrix: */
	ndim = atoi ( argv[ 1 ] ) ;
	/* "Enter the number of threads: */
	NUM_THREADS = atoi ( argv[ 2 ] ) ;
	double p ;

	printf ("The dimension is %d and No. of Threads are: %d \n" , ndim , NUM_THREADS );

	/*
	Create matrix on heap by malloc and storing and assigning pointers to
	the matrix as matA.
	The matrix is a linear array of size ndim x ndim containing the coefficients of linear equations
	Total memory malloced is ndim^2
	Matrix B is a vector of size ndim used as the values of the linear equations
	 */

	double *matA [ ndim ] ; /* Matrix A (ndim x ndim) */
	double *matA_check [ ndim ] ; /*Deepcopy of Matrix A (ndim x ndim)*/
	double *I [ ndim ] ; /* Identity matrix (ndim x ndim) */
	double *I_check [ ndim ] ; /* Identity matrix (ndim x ndim) */
	double *x [ ndim ] ; /* Matrix X (ndim x ndim) */
	double *matAx [ ndim ] ; /* Matrix Multiplication of matA_check and X (ndim x ndim) */

	for ( int i = 0 ; i < ndim ; i++ )
	{
		matA [ i ] = (double *)malloc( ndim * sizeof(double) );
		matA_check [ i ] = (double *)malloc( ndim * sizeof(double) );
		I [ i ] = (double *)malloc( ndim * sizeof(double) );
		I_check[ i ] = (double *)malloc( ndim * sizeof(double) );
		x [ i ] = (double *)malloc( ndim * sizeof(double) );
		matAx [ i ] = (double *)malloc( ndim * sizeof(double) );
	}
	/* Populate matrix A and B with random numbers */
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
			matA_check[ i ][ j ] = matA[ i ][ j ] ;
			matAx[ i ][ j ] = 0.0 ;
			if ( i == j )
			{
				I [ i ][ j ] = 1.0 ;
				I_check [ i ][ j ] = 1.0 ;
			}
			else
			{
				I [ i ][ j ] = 0.0 ;
				I_check [ i ][ j ] = 0.0 ;
			}
		}
	}
/*---------------------------------------------------------------*/
	chunk = ( int ) ndim / NUM_THREADS ; /*Initiates the chunk size for the work distribution*/
	// Start high resolution clock timer
	clock_gettime( CLOCK_MONOTONIC , &start );
	for ( int i = 0 ; i < ndim - 1 ; i++ )
	{
		/*Start Triangularization*/
		find_max ( matA , i , ndim , I ) ; /*Finds the max for partial pivoting*/
		omp_set_num_threads( NUM_THREADS ); /*Set number of threads to be used in the parallel section ahead*/
		#pragma omp parallel shared( matA , I , i , chunk , ndim ) private( p , j , k ) /*Start Parallel section of code*/
		{
			#pragma omp for schedule(static,chunk) /*Divide the for loop using the omp for with static work distribution*/
				for ( j = i + 1 ; j < ndim ; j++ )
				{
					p = matA[ j ][ i ] / matA[ i ][ i ] ;
					for ( k = 0 ; k < i ; k++ )
					{
						I [ j ][ k ] -= p * ( I[ i ][ k ] ) ;
					}
					for ( int k = i ; k < ndim ; k++ )
					{
						matA[ j ][ k ] -= p * ( matA[ i ][ k ] ) ;
						I [ j ][ k ] -= p * ( I[ i ][ k ] ) ;
					}
				}
			#pragma omp barrier /*Use barrier for waiting for the threads to join, before execution goes to next iteration*/
		}
	}
		/*Start Backsubstitution*/
		for ( int i = ndim - 1 ; i >= 0 ; i-- )
		{
			for ( int j = 0 ; j < ndim ; j++ )
			{
				x[ i ][ j ] = I[ i ][ j ] / matA[ i ][ i ] ;
			}
			omp_set_num_threads( NUM_THREADS ); /*Set number of threads to be used in the parallel section ahead*/
			#pragma omp parallel shared( matA , x , I , i , chunk , ndim ) private( k , l ) /*Start Parallel section of code*/
			{
				#pragma omp for schedule(dynamic,chunk) /*Divide the for loop using the omp for with static work distribution*/
				for ( int k = 0 ; k < i ; k++ )
				{
					for ( l = 0 ; l < ndim ; l++ )
					{
						I [ k ][ l ] -= matA[ k ][ i ] * x[ i ][ l ] ; 
					}
				}
				#pragma omp barrier /*Use barrier for waiting for the threads to join, before execution goes to next iteration*/
			}
		} 

	/* Calculate the execution time */
	clock_gettime( CLOCK_MONOTONIC , &end );	/* End clock timer. */
	/* Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9 */
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "Elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	/*Do error checking using residuals of the matrix*/
	double sum_r = 0.0 , sum_a = 0.0 , sum_x = 0.0 ;
	double err ;


	//Iterate through the rows of A and X
	for ( int i = 0 ; i < ndim ; i++ )
	{
		// Iterate through the columns B and C
		for ( int j = 0; j < ndim ; j++ )
		{
			// Iterate through the columns of A and Rows of B
			for ( int k = 0 ; k < ndim ; k++)
			{
				// Multiply the two elements
				matAx[ i ][ j ] += matA_check[ i ][ k ] * x[ k ][ j ] ;
			}
		}
	}

	for ( int i = 0 ; i < ndim ; i++ )
	{
		for ( int j = 0 ; j < ndim ; j++ )
		{
			sum_r += pow( matAx[ i ][ j ] - I_check[ i ][ j ] , 2 ) ;
			sum_a += pow ( matA_check[ i ][ j ] , 2 ) ;
			sum_x += pow ( x[ i ][ j ] , 2 ) ;
		}
	}
	err = pow( sum_r / ( sum_a * sum_x ) , 0.5 ) ;
	printf ("Error is %lf\n", err);


	/*Deallocate the memory allocated to matrices A, B, A_check, B_check and x */
	for ( int i = 0 ; i < ndim ; i++ ) 
	{ 
		free( matA[ i ] );
		free( matA_check[ i ] );
		free( I[ i ] ) ;
		free( I_check[ i ] ) ;
		free( x[i] ) ;
		free( matAx[ i ] ) ;
	}
	exit( 0 ) ;
}