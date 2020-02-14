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

#define BILLION 1000000000L	  // To convert clock time in floating point seconds to nanoseconds/
int NUM_THREADS ;	  // No. threads will be created

typedef struct 
{
double *matA ;
double *matB ;
int ndim ;
int     veclen ; 
int i , j , thr_id;
float p ;
} STRUCT_ARGS ;
STRUCT_ARGS str_args ;
pthread_mutex_t mutexsum ; 

void* gaussian_elimination ( void* arg )
{
	STRUCT_ARGS *str_args;
	str_args = ( STRUCT_ARGS * ) arg ;
	int ndim = str_args -> ndim ;
	double *matA = str_args -> matA ;
	double *matB = str_args -> matB ;
	int veclen = str_args -> veclen ;
	int i = str_args -> i ;
	int j = str_args -> j ;
	long thr_id = str_args -> thr_id ;
	int start = thr_id * veclen ;

	for ( int k = i + veclen * thr_id ; k < i + veclen  * ( thr_id + 1 ) ; k++ )
	{
		*( matA + j * ndim + k ) -= str_args -> p * ( *( matA + i * ndim + k ) ) ;
	}
	pthread_exit( (void * ) (long)thr_id  );
}

void* back_substitution ( void* arg)
{	
	STRUCT_ARGS *str_args;
	str_args = ( STRUCT_ARGS * ) arg ;
	int ndim = str_args -> ndim ;
	double *matA = str_args -> matA ;
	double *matB = str_args -> matB ;
	int veclen = str_args -> veclen ;
	int i = str_args -> i ;
	int start = str_args -> thr_id * veclen ;
	int thr_id = str_args -> thr_id ;
	float x = str_args -> p;
	for ( int k = thr_id * veclen ; k < ( thr_id + 1 ) * veclen , k <= i ; k++ )
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
	int num;
	float p , x;
	int rc , attr_st;
    void *status;

	printf( "Enter the dimension of the matrix:\n" );
	scanf("%d" , &ndim); 	// Store matrix dimension in ndim 
	printf( "Enter the number of threads:\n" );
	scanf("%d" , &NUM_THREADS); 	// Store matrix dimension in ndim 	

	str_args . ndim = ndim ;
	str_args . veclen = ( int ) ndim / NUM_THREADS ;
	str_args . i = 0 ;
	str_args . j = 0 ;
	/*
	Create matrix on heap by malloc and storing and assigning pointers to
	the matrix as matA.
	The matrix is a linear array of size ndim x ndim
	Total memory malloced is ndim^2
	 */
	double* matA = ( double * )malloc( ndim * ndim * sizeof( double ) ) ;
	double* matB = ( double * )malloc( ndim * sizeof( double ) ) ;
	str_args . matA = matA ;
	str_args . matB = matB ;
	 
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
			pthread_t tids[ NUM_THREADS ];
			pthread_attr_t attr;	
			pthread_attr_init(&attr);
    		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
			for (int id = 0; id < NUM_THREADS; id++) 
			{
				str_args . thr_id = id ;
				str_args . i = i ;
				str_args . j = j ;
				str_args . p = p ;
				pthread_create( &tids[id] , &attr , gaussian_elimination , ( void* ) &str_args );
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
	
	
	for ( int i = ndim - 1 ; i >= 0 ; i-- )
	{
		if ( * (matA + i * ndim + i) == 0 )
		{
			printf( "No Solution Exists" ) ;
			exit( -1 ) ;
		}

		x = *( matB + i ) / * (matA + i * ndim + i) ;
		printf ( "Answer: %f \n" , x ) ; 
		pthread_t tids[ NUM_THREADS ];
		pthread_attr_t attr;	
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		for (int id = 0; id < NUM_THREADS; id++) 
		{
			str_args . thr_id = id ;
			str_args . i = i ;
			str_args . p = x ;
			// pthread_attr_init( &attr );
			pthread_create( &tids[id] , &attr , back_substitution , ( void* ) &str_args );
		}
		pthread_attr_destroy(&attr);
		for (int id = 0; id < NUM_THREADS; id++) 
		{
			pthread_join(tids[id], NULL);
		}
		
	}	
	
	clock_gettime( CLOCK_MONOTONIC , &end );	// End clock timer.
	//Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );

	//Deallocate the memory allocated to matrices A, B and C
	free ( matA ) ;
	free ( matB ) ;
	pthread_exit(NULL);
	exit( 0 ) ;
}