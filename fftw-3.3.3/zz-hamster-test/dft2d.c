# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <mpi.h>
# include <fftw3-mpi.h>

 int main(int argc, char **argv)
 {
     const ptrdiff_t L = 1024, M = 1024;
     fftw_plan plan;
     fftw_complex *data ;
     ptrdiff_t alloc_local, local_L, local_L_start, ii;
     int i, j, ierr, nproc, myid;
     double xx, yy, rr, r2, t0, t1, t2, t3, tplan, texec;
     const double amp = 256;
     /* Initialize */
     MPI_Init(&argc, &argv);
     fftw_mpi_init();
     ierr = MPI_Comm_size(MPI_COMM_WORLD, &nproc);
     ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);

     /* get local data size and allocate */
     alloc_local = fftw_mpi_local_size_2d(L, M, MPI_COMM_WORLD, &local_L, &local_L_start);
     data = fftw_alloc_complex(alloc_local);
     /* create plan for in-place forward DFT */
     t0 = MPI_Wtime();
     plan = fftw_mpi_plan_dft_2d(L, M, data, data, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
     t1 = MPI_Wtime();
     /* initialize data to some function my_function(x,y) */
     for (i = 0; i < local_L; ++i) for (j = 0; j < M; ++j)
     {
    ii = i + local_L_start;
        xx = ( (double) ii - (double)L/2 );
        yy = ( (double)j - (double)M/2 );
        r2 = pow(xx, 2) + pow(yy, 2);
        rr = sqrt(r2);
        if (rr <= amp)
        {
          data[i*M + j][0] = 1.;
          data[i*M + j][1] = 0.;
        }
        else
        {
          data[i*M + j][0] = 0.;
          data[i*M + j][1] = 0.;
        }
     }
     /* compute transforms, in-place, as many times as desired */
     t2 = MPI_Wtime();
     fftw_execute(plan);
     t3 = MPI_Wtime();
     /* Print results */
     tplan = t1 - t0;
     texec = t2 - t1;
     printf(" T_plan = %f, T_exec = %f \n",tplan,texec);
     /* deallocate and destroy plans */
     fftw_destroy_plan(plan);
     fftw_mpi_cleanup();
     fftw_free ( data );
     MPI_Finalize();
 }
