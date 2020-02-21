/*
Author: Vedanta Pawar
NetID: vp273
Class: M.Eng ECE, Cornell University
Email: vp273@cornell.edu

Instructions for Compiling and Executing Code:
Compile: gcc vp273_hw2_code.c -o vp273_hw2_code -std=gnu99 -pthread -lm
Run: ./vp273_hw2_code "Enter the dimension of the matrix:" "Enter the number of threads:"
Example: ./vp273_hw2_code 4096 8
*/

#include <stdio.h> 	 /* For printf()  */
#include <stdint.h>	 /* For uint64 */
#include <stdlib.h>  /* For srand48() and drand48() */
#include <time.h>    /* For clock_gettime() */
#include <pthread.h> /* For pthreads */
#include <math.h> 	 /* For verification of results */

#define BILLION 1000000000L	  /* To convert clock time in floating point seconds to nanoseconds */
int NUM_THREADS ;	  		  /* No. threads will be created */

struct STRUCT_ARGS /* For Passing Arguments via the pthread_create function */
{
double *matA ; /*  Matrix A contains the coefficients of the x of a set of linear equations */
double *matB ; /*  Matrix B contains the values of the linear equations */
int ndim ;     /* Dimensions of A such that ndim * ndim */
int veclen ;   /* Each thread is alloted a set of rows for performing elementary row operations. Every thread is alloted
				ndim / NUM_THREADS */
int i ; 	   /* Current row to perform the pivoting */
int thr_id ;   /* Thread ID used for alloting the rows to work upon */
double x ;     /* Value of x used in back substitution */
}  ;

/* Performs triangularization on the matrix A
using pthreads. Each thread is alloted a set of rows
for performing elementary row operations. Every thread is alloted
ndim / no_of_threads, where ndim is the dimension of the matrix
 */
void* triangularization ( void* arg )
{
	struct STRUCT_ARGS *str_args;
	str_args = ( struct STRUCT_ARGS * ) arg ;
	int ndim = str_args -> ndim ;
	double *matA = str_args -> matA ;
	double *matB = str_args -> matB ;
	int veclen = str_args -> veclen ;
	int i = str_args -> i ;
	double p ;	  /* Store the value for the scalar multiplier used for row operations */
	long thr_id = str_args -> thr_id ;

	/* Allot rows according to the thread id. */
	for ( int j = i + 1 + thr_id * veclen ; j < i + 1 + ( thr_id + 1 ) * veclen && j < ndim ; j++ )
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
	pthread_exit( (void * ) (long)thr_id  );  /* Pthread exit and join the main thread*/
}


/* The function back_substitution is used on the traingula matrix A
and calculates the x for every row. Every thread is alloted the number of
rows according to ndim / NUM_threads
 */
void* back_substitution ( void* arg)
{	
	struct STRUCT_ARGS *str_args;
	str_args = ( struct STRUCT_ARGS * ) arg ;

	int ndim = str_args -> ndim ;
	double *matA = str_args -> matA ;
	double *matB = str_args -> matB ;
	int veclen = str_args -> veclen ;
	int i = str_args -> i ;
	int start = str_args -> thr_id * veclen ;
	int thr_id = str_args -> thr_id ;
	double x = str_args -> x;

	/* Performs backsubstitution on the  */
	for ( int k = thr_id * veclen ; k < ( thr_id + 1 ) * veclen && k <= i ; k++ )
	{
		*( matB + k ) -= x * ( *( matA + k * ndim + i ) ) ; 
	}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] )
{
	uint64_t diff; 				/* Stores the time in nanoseconds */
	struct timespec start, end; /* Used to implement the high resolution timer included in <time.h> */
	int ndim ;					/* Ask user and store the dimension of the square matrix */
	int rc , attr_st ;          /* Used for error checking of the pthread_join return value */
    void *status ;				/* Used for error checking of the pthread_join return value for debugging purposes. NOTE: not used for now*/
	double sum = 0.0 ;			/* Checking the sum of Gaussian elimination */
	double rho = 0.0 ;			/* Finding the inner product of the error residuals  */

	/* "Enter the dimension of the matrix: */
	ndim = atoi ( argv[ 1 ] ) ;
	/* "Enter the number of threads: */
	NUM_THREADS = atoi ( argv[ 2 ] ) ;

	printf ("The dimension is %d and No. of Threads are: %d \n" , ndim , NUM_THREADS );

	/*
	Create matrix on heap by malloc and storing and assigning pointers to
	the matrix as matA.
	The matrix is a linear array of size ndim x ndim containing the coefficients of linear equations
	Total memory malloced is ndim^2
	Matrix B is a vector of size ndim used as the values of the linear equations
	 */
	double *matA = ( double * )malloc( ndim * ndim * sizeof( double ) ) ;
	double *matB = ( double * )malloc( ndim * sizeof( double ) ) ;
	double *matA_check = ( double * )malloc( ndim * ndim * sizeof( double ) ) ;
	double *matB_check = ( double * )malloc( ndim * sizeof( double ) ) ;
	double* x = ( double * )malloc( ndim * sizeof( double ) ) ;
	struct STRUCT_ARGS str_args[NUM_THREADS] ;

	/* Populate matrix A and B with random numbers */
	clock_gettime( CLOCK_MONOTONIC , &start );
	for ( int i = 0 ; i < ndim ; i++ )
	{
		/* Iterate through the columns of the Matrix A  */
		for ( int j = 0 ; j < ndim ; j++ )
		{
			clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
			/* Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9 */
			diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
			srand48( diff ) ; /* Set random seed to for initializing drand48() later */
			/* Store some random numbers in A */
			*( matA + i * ndim + j ) = drand48() ;		
			*( matA_check + i * ndim + j ) = *( matA + i * ndim + j ) ;
		}
		clock_gettime( CLOCK_MONOTONIC , &end );	/* End clock timer. */
		/* Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9 */
		diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
		srand48( diff ) ; /* Set random seed to for initializing drand48() later*/
		*( matB + i ) = drand48() ;	
		* ( matB_check + i ) = *( matB + i ) ;
	}

	/* Perform Gaussian elimination. First we perform traingularization
	   Start high resolution clock timer */
	clock_gettime( CLOCK_MONOTONIC , &start );
	for ( int i = 0 ; i < ndim - 1 ; i++ )
	{
		pthread_t tids[ NUM_THREADS ];  /* Create Pthread IDs*/
		/* Allot work to every thread */
		for (int id = 0; id < NUM_THREADS; id++) 
		{
			pthread_attr_t attr;	
			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);	/* Create Joinable Thread */
			/* Pass arguments in as structure to be passed to the thread */
			str_args[id] . matA = matA ;
			str_args[id] . matB = matB ;
			str_args[id] . thr_id = ( long ) id ;
			str_args[id] . i = i ;
			str_args[id] . ndim = ndim ;
			str_args[id] . veclen = (int)ceil( (float)( ndim - i )/ NUM_THREADS ) ;
			pthread_create( &tids[id] , &attr , triangularization , ( void* ) &str_args[id] );
		} 
		/* Wait for the threads to join */
		for (int id = 0; id < NUM_THREADS; id++) 
		{
			rc = pthread_join(tids[id], &status);
			if (rc) 
			{
				printf("ERROR; return code from pthread_join() is %d and id is %d \n", rc , id);
				exit(-1);
			}
		}	
	}
/*
##########################################################
_____________________BackSubstitution_____________________
##########################################################
*/

	for ( int i = ndim - 1 ; i >= 0 ; i-- )
	{
		if ( * (matA + i * ndim + i) == 0.0 )
		{
			*( x + i ) = 0.0 ;  /* If x[i] = 0 then rank is ndim - 1 */
		}
		else 
		{
			*( x + i ) = *( matB + i ) / * (matA + i * ndim + i) ;
			/* printf ( "Answer: %lf \n" , *( x + i ) ) ; // Print the values of x */
			pthread_t tids[ NUM_THREADS ];
			/* Allot work to every thread */
			for (int id = 0; id < NUM_THREADS; id++) 
			{
				pthread_attr_t attr;	
				pthread_attr_init(&attr);
				pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
				str_args[id] . matA = matA ;
				str_args[id] . matB = matB ;
				str_args[id] . thr_id = ( long ) id ;
				str_args[id] . i = i ;
				str_args[id] . x = *( x + i ) ;
				str_args[id] . ndim = ndim ;
				str_args[id] . veclen = (int)ceil( (float)ndim / NUM_THREADS ) ;
				pthread_create( &tids[id] , &attr , back_substitution , ( void* ) &str_args[id] );
			}
			for (int id = 0; id < NUM_THREADS; id++) 
			{
				rc = pthread_join(tids[id], &status);
				if (rc) 
				{
					printf("ERROR; return code from pthread_join() is %d and id is %d \n", rc , id);
					exit(-1);
				}
			}
		}
	}	
	
	/* Calculate the execution time */
	clock_gettime( CLOCK_MONOTONIC , &end );	/* End clock timer. */
	/* Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9 */
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "Elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	/* Perform Numerical Verfication and error calculation */
	printf ( "___________Numerical Verification___________\n" ) ;
	for ( int i = 0 ; i < ndim ; i++ )
	{
		sum = 0.0 ;
		for ( int j = 0 ; j < ndim ; j++ )
		{
			sum += *( matA_check + i * ndim + j ) * ( *( x + j ) ) ;
		}
		rho += pow ( ( sum - *(matB_check + i ) ) , 2 )  ;
	}
	rho = sqrt ( rho ) ;
	printf ( "The error is %lf\n" , rho ) ;

	/*Deallocate the memory allocated to matrices A, B, A_check, B_check and x */
	free ( matA ) ;
	free ( matB ) ;
	free ( matA_check ) ;
	free ( matB_check ) ;
	free ( x ) ;
	pthread_exit(NULL);
	exit( 0 ) ;
}