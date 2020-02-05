/*
Author: Vedanta Pawar
Class: M.Eng ECE, Cornell University
Email: vp273@cornell.edu
*/
// http://www.netlib.org/utk/papers/autoblock/node2.html

#include <stdio.h>
#include <stdint.h>	
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define BILLION 1000000000L

int main( int argc, char *argv[] )
{
	uint64_t diff;
	struct timespec start, end;
	int ndim , tile_size , N ;
	float sum = 0.0;
	float ele;

	printf( "Enter the dimension of the matrix to be multiplied:\n" );
	scanf("%d" , &ndim);
	printf( "Enter the tile size, note that the %d ndim dimension should be divisible by tile size:\n" , ndim );
	scanf("%d" , &tile_size);    
    assert ( ndim % tile_size == 0 );
	
	double *matA = ( double * )malloc( ndim * ndim * sizeof( double ) );
	double *matB = ( double * )malloc( ndim * ndim * sizeof( double ) );
	double *matC = ( double * )malloc( ndim * ndim * sizeof( double ) );
	srand48 (1);	

	for ( int i = 0 ; i < ndim ; i++ )
	{
		for ( int j = 0 ; j < ndim ; j++ )
		{
			*( matA + i * ndim + j ) = drand48() ;
            // scanf("%f" , &ele);
			// *( matA + i * ndim + j ) = ele ;
		}
	}		

	for ( int i = 0 ; i < ndim ; i++ )
	{
		for ( int j = 0 ; j < ndim ; j++ )
		{
			*( matB + i * ndim + j ) = drand48() ;
            // scanf("%f" , &ele);
			// *( matB + i * ndim + j ) = ele ;
		}
	}

    N = ( int ) ( ndim / tile_size ) ; // Number of Blocks
	clock_gettime( CLOCK_MONOTONIC , &start );

    for ( int i0 = 0 ; i0 < N ; i0++ )
    {
        for ( int j0 = 0 ; j0 < N ; j0++ )
        {
            for ( int k0 = 0 ; k0 < N ; k0++ )
            {
                for ( int i = i0; i < i0 + tile_size ; i++ )
                {
                    for ( int j = j0 ; j < j0 + tile_size ; j++)
                    {
                        for ( int k = k0 ; k < k0 + tile_size ; k++ )
                        {
                            *( matC + i * ndim + j ) += *( matA + i * ndim + k ) * ( *( matB + k * ndim + j ) ) ;
                        }
                    }			
                }
            }
        }

    }
	
    // for ( int i = 0 ; i < ndim ; i++ )
	// {
	// 	for ( int j = 0 ; j < ndim ; j++ )
	// 	{
	// 		printf("eleAB %f \n" , *( matC + i * ndim + j ) );
	// 	}
	// }
	clock_gettime( CLOCK_MONOTONIC , &end );
	diff = BILLION * ( end.tv_sec - start.tv_sec ) + end.tv_nsec - start.tv_nsec;
	printf( "elapsed time = %llu nanoseconds\n", ( long long unsigned int ) diff );
	free ( matA ) ;
	free ( matB ) ;
	free ( matC ) ;
        
	exit( 0 ) ;
}