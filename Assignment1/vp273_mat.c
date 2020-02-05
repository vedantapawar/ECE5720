/*
Author: Vedanta Pawar
NetID: vp273
Class: M.Eng ECE, Cornell University
Email: vp273@cornell.edu

Instructions for Compiling and Executing Code:
Compile: gcc vp273_mat.c -std=gnu99 -o vp273_mat
Run: ./vp273_mat
*/

#include <stdio.h> 	// For printf() 
#include <stdint.h>	// For uint64
#include <stdlib.h> // For exit(0)
#include <time.h>   // For clock_gettime()

#define MAX_LENGTH 1048576  // MAX_LENGTH is 2^20
#define STRIDE 1            // Stride length 
#define MIN_SIZE 1024       // MIN_LENGTH is 2^10
#define BILLION 1000000000L // To convert clock time in floating point seconds to nanoseconds
#define REPEAT 10           // For averaging out the result over 10 trials

int main( int argc, char *argv[] )
{
    float* A;                   // Pointer to the array
    uint64_t diff;              // Stores the time in nanoseconds
	struct timespec start, end; // Used to implement the high resolution timer included in <time.h>
    double avg_time;            // Store the average time for a single access
    FILE *ptr_file ;            // File pointer used to point to a .csv file for storing the time vs stride

    /* Malloc an array of size MAX_LENGTH x size of float and 
    assign the pointer to A */
    A = ( float* )malloc( MAX_LENGTH * sizeof( float ) );

    if ( A == NULL ) // If memory could no be allocated then throw error message
    {
        printf( "Memory not allocated.\n" ); 
        exit(0); 
    }
    
    ptr_file = fopen("output.csv", "w");  // Save the time and corresponding stride in a csv file

    // Iterate through the different array length. n:=2*n
    for ( int n = MIN_SIZE; n <= MAX_LENGTH ; n = 2 * n )
    {
        // Iterate through different stride length. s:=2*n
    	for ( int s = 1 ; s <= n / 2 ; s = 2 * s )
    	{
            // Start high resolution clock timer
    		clock_gettime( CLOCK_MONOTONIC , &start );
            //Repeat the array access REPEAT x Stride times
    		for ( int j = 0 ; j < REPEAT * s ; j++ )
    		{
                // Access("touch") the array n / s times
    			for ( int i ; i <= (int)( n / s ) ; i++ )
	    		{
	    			A[ i * s ] = 3.142857 ; // Touch the array element by storing pi
	    		}
    		}            
    		clock_gettime( CLOCK_MONOTONIC , &end );    // End clock timer.
            //Calculate the difference in timer and convert it to nanosecond by multiplying by 10^9
    		diff = ( BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec ) ;
            // Calculate average time for a single access by dividing by n x REPEAT
            avg_time = ( double )diff / ( n * REPEAT ) ;
			printf( "elapsed time = %lf nanoseconds for STRIDE= %d and LENGTH= %d\n", avg_time , s , n);
            // Store the access time and corresponding stride length in a csv file
            fprintf( ptr_file ,"%d , %lf\n", s ,  avg_time );
    	}
    }   
    free ( A ) ;            // Free allocated memory pointed by A
    fclose ( ptr_file ) ;   // Close the .csv file
    exit( 0 ) ;
}