/*
	Author : Sai Ganesh
	Roll No: 167101;
	BTECH CSE NIT WARANGAL 

	Implementation of Cannon's Algorithm
*/

#include	<stdio.h>
#include	<stdlib.h>
#include	<unistd.h>
#include	<mpi.h>
#include	<math.h>


/* Prototype Declaration for Function */
void matrixMultiply(int, double *, double *, double *);

int main(int argc, char **argv)
{
	int i;
	int n = 2;
	int nlocal;
	int npes, dims[2], periods[2];
	int myrank, my2drank, mycoords[2];
	int uprank, downrank, leftrank, rightrank, coords[2];
	int shiftsource, shiftdest;
	MPI_Status status;
	MPI_Comm comm_2d, comm;
	comm = MPI_COMM_WORLD;

	MPI_Init(&argc,&argv);

	/*Get the communicator related information */
	MPI_Comm_size(comm, &npes);
	MPI_Comm_rank(comm, &myrank);

	/*Set up the Cartesian Topology */
	dims[0] = dims[1] = sqrt((double)npes);

	/*Set Periods for Wrap Around Condition */
	periods[0] = periods[1] = 1;


	/*	Create Cartesian Topology with rank reordering 
		
		Parameters to the Function are :

		* Communication group
		* No. of Dimensions 
		* Dimensions Array
		* Periods array ( i.e., for wrap around connections )
		* reorder = 1 ( imp. to match the underlying physical topology)
		* New Communication Group
	*/

	MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);

	/*Get the rank and co-ordinates w.r.t the new topology */
	MPI_Comm_rank(comm_2d, &my2drank);
	MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);
	
	
	/*Compute Ranks of the up and left shift */
	MPI_Cart_shift(comm_2d, 1, -1, &rightrank, &leftrank);		//Along Dimension 1 (Rows)
	MPI_Cart_shift(comm_2d, 0, -1, &downrank, &uprank);		//Along Dimension 0  (Columns)


	/*Determine the dimension of the Local Block matrix */
	nlocal = n/dims[0];

	/*
		Setup Local Buffers of a, b, c
	*/

	double *a = (double *)malloc(nlocal*nlocal*sizeof(double));
	double *b = (double *)malloc(nlocal*nlocal*sizeof(double));
	double *c = (double *)malloc(nlocal*nlocal*sizeof(double));

	for (int i = 0; i < nlocal; ++i)
	{
		for (int j = 0; j < nlocal; ++j)
		{
			a[i*nlocal + j] = (double)((rand() + my2drank) % 50);
			b[i*nlocal + j] = (double)((rand() + my2drank) % 50);
			c[i*nlocal + j] = 0.0;
		}
	}
	
	/*Perform the initial Alignment for A and then for B */
	
	/*Shift each row of matrix A by i steps to left (with wrapAround ) */
	MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest);

	MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, shiftdest,
										1, shiftsource, 1, comm_2d, &status);

	/*shift each column of matrix B by j steps upward	(with wrapAround) */
	MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest);

	MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE, shiftdest,
										1, shiftsource, 1, comm_2d, &status);

	/*Get into Main Computation Loop */
	for (int i = 0; i < dims[0]; ++i)
	{

      	matrixMultiply(nlocal,a,b,c);

    	/*Shift Matrix A left by one */
		MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, leftrank,
										1, rightrank, 1, comm_2d, &status);

		/*Shift Matrix B Upward by one */

		MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE, uprank, 
										1, downrank, 1, comm_2d, &status);	
        
	}

	MPI_Comm_free(&comm_2d);

	MPI_Finalize();

	return 0;
}

void matrixMultiply(int n, double *a, double *b, double *c)
{

	//C[i][j] = 0 for all i, j in (0,n)


	/* 
		The below matrix multiplication(i, k, j) is optimised version of the
			the known (i, j ,k)

		This below version uses the L1 cache more efficiently 
	*/
	
	for (int i = 0; i < n; ++i)
	{
		for (int k = 0; k < n; ++k)
		{
			for (int j = 0; j < n; ++j)
			{
				c[i*n + j] += a[i*n + k]*b[k*n + j];
			}
		}
	}
}
