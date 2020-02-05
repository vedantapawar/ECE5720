/*
Author: Vedanta Pawar
Class: M.Eng ECE, Cornell University
Email: vp273@cornell.edu
*/

#include <stdio.h>
#include <stdint.h>	
#include <stdlib.h>
#include <time.h>

#define MAX_LENGTH 8388608
#define STRIDE 1
#define MIN_SIZE 1024
#define BILLION 1000000000L
#define REPEAT 10

int main( int argc, char *argv[] )
{
    uint64_t diff;
	struct timespec start, end;
    float* A;
    FILE *ptr_file ;

    A = ( float* )malloc( MAX_LENGTH * sizeof( float ) );
    if ( A == NULL )
    {
        printf( "Memory not allocated.\n" ); 
        exit(0); 
    }
    
    ptr_file = fopen("output.csv", "w");
    for ( int n = MIN_SIZE; n <= MAX_LENGTH ; n = 2 * n )
    {
    	for ( int s = 1 ; s <= n / 2 ; s = 2 * s )
    	{
    		clock_gettime( CLOCK_MONOTONIC , &start );
    		for ( int j = 0 ; j < REPEAT ; j++ )
    		{
    			for ( int i ; i <= (int)( n / s ) ; i++ )
	    		{
	    			A[ i * s ] = 3.142857 ;
	    		}
    		}
    		clock_gettime( CLOCK_MONOTONIC , &end );
    		diff = ( BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec ) ;
			printf( "elapsed time = %lf nanoseconds for STRIDE= %d and LENGTH= %d\n", ( double ) diff * s / ( n * REPEAT) , s , n);
            fprintf( ptr_file ,"%d , %lf\n", s ,  ( double ) diff * s / ( n * REPEAT ) );
    	}
    }   
    free ( A ) ;
    fclose ( ptr_file ) ;
    exit( 0 ) ;
}