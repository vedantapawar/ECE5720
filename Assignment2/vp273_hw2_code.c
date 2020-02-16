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
#include <pthread.h>
#include <math.h>

#define BILLION 1000000000L	  // To convert clock time in floating point seconds to nanoseconds/
int NUM_THREADS ;	  // No. threads will be created

struct STRUCT_ARGS
{
double *matA ;
double *matB ;
int ndim ;
int veclen ; 
int i , j , thr_id;
float p ;
}  ;


void* gaussian_elimination ( void* arg )
{
	struct STRUCT_ARGS *str_args;
	str_args = ( struct STRUCT_ARGS * ) arg ;
	int ndim = str_args -> ndim ;
	double *matA = str_args -> matA ;
	double *matB = str_args -> matB ;
	int veclen = str_args -> veclen ;
	int i = str_args -> i ;
	int j = str_args -> j ;
	long thr_id = str_args -> thr_id ;

	for ( int k = i + veclen * thr_id ; k < i + veclen  * ( thr_id + 1 ) && k < ndim ; k++ )
	{
		// printf("%lf , %lf , i=%d , j=%d , k=%d , tid=%ld\n" , *( matA + j * ndim + k ) , *( matA + i * ndim + k ) , 
		// i , j , k ,thr_id );
		*( matA + j * ndim + k ) -= str_args -> p * ( *( matA + i * ndim + k ) ) ;
	}
	pthread_exit( (void * ) (long)thr_id  );
}

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
	double x = str_args -> p;
	for ( int k = thr_id * veclen ; k < ( thr_id + 1 ) * veclen && k <= i ; k++ )
	{
		*( matB + k ) -= x * ( *( matA + k * ndim + i ) ) ; 
	}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] )
{
	uint64_t diff; 				// Stores the time in nanoseconds
	struct timespec start, end; // Used to implement the high resolution timer included in <time.h>
	int ndim ;					// Ask user and store the dimension of the square matrix
	int num ;
	double p ;
	int rc , attr_st ;
    void *status ;
	double sum = 0.0 ;
	double rho = 0.0 ;

	printf( "Enter the dimension of the matrix:\n" );
	scanf("%d" , &ndim); 	// Store matrix dimension in ndim 
	printf( "Enter the number of threads:\n" );
	scanf("%d" , &NUM_THREADS); 	// Store matrix dimension in ndim 	

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
	struct STRUCT_ARGS str_args[NUM_THREADS] ;

	// Iterate through the rows of the Matrix A 
	for ( int i = 0 ; i < ndim ; i++ )
	{
		// Iterate through the columns of the Matrix A 
		for ( int j = 0 ; j < ndim ; j++ )
		{
			srand48( clock() ) ; // Set random seed to for initializing drand48() later
			// Store same random numbers in A 
			*( matA + i * ndim + j ) = drand48()  ;		
			// scanf( "%d" , &num )	;
			// *( matA + i * ndim + j ) = num ;	
			*( matA_check + i * ndim + j ) = *( matA + i * ndim + j ) ;
	
		}
		srand48 ( clock() ) ;
		*( matB + i ) = drand48() ;	
		// scanf( "%d" , &num )	;
		// *( matB + i ) = num ;	
		* ( matB_check + i ) = *( matB + i ) = num ;
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
			pthread_t tids[ NUM_THREADS ];
			for (int id = 0; id < NUM_THREADS; id++) 
			{
				pthread_attr_t attr;	
				pthread_attr_init(&attr);
				pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);	
				str_args[id] . matA = matA ;
				str_args[id] . matB = matB ;
				str_args[id] . thr_id = ( long ) id ;
				str_args[id] . i = i ;
				str_args[id] . j = j ;
				str_args[id] . p = p ;
				str_args[id] . ndim = ndim ;
				str_args[id] . veclen = (int)ceil( (float)ndim / NUM_THREADS ) ;
				pthread_create( &tids[id] , &attr , gaussian_elimination , ( void* ) &str_args[id] );
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
			*( matB + j ) -= p * ( *( matB + i ) ) ;
		}
	}
/*
##########################################################
_____________________BackSubstitution_____________________
##########################################################
*/

	for ( int i = ndim - 1 ; i >= 0 ; i-- )
	{
		if ( * (matA + i * ndim + i) == 0 )
		{
			printf( "No Solution Exists" ) ;
			*( x + i ) = 0.0 ;
		}

		else 
		{
			*( x + i ) = *( matB + i ) / * (matA + i * ndim + i) ;
		}
		
		printf ( "Answer: %lf \n" , *( x + i ) ) ; 
		pthread_t tids[ NUM_THREADS ];
		for (int id = 0; id < NUM_THREADS; id++) 
		{
			pthread_attr_t attr;	
			pthread_attr_init(&attr);
			pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
			str_args[id] . matA = matA ;
			str_args[id] . matB = matB ;
			str_args[id] . thr_id = ( long ) id ;
			str_args[id] . i = i ;
			str_args[id] . p = *( x + i ) ;
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
	
	clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
	//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	for (int i = 0; i < ndim; i++)
	{
		for (int j = 0; j < ndim; j++)
		{
			printf("MatA %lf\n" , *( matA + i * ndim + j )) ;
		}
	}
	

	printf ( "___________Numerical Verification___________\n" ) ;
	for ( int i = 0 ; i < ndim ; i++ )
	{
		sum = 0.0 ;
		for ( int j = 0 ; j < ndim ; j++ )
		{
			sum += *( matA_check + i * ndim + j ) * ( *( x + j ) ) ;
			// printf("%lf , %lf \n" , *( matA_check + i * ndim + j ) , ( *( x + j )) );
		}
		// printf("%lf , %lf \n" , sum , *(matB_check + i ) );
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
	pthread_exit(NULL);
	exit( 0 ) ;
}