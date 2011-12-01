/******************************************************************************
* Copyright (C) 2011 Technische Universitaet Muenchen                         *
* This file is part of the training material of the master's course           *
* Scientific Computing                                                        *
******************************************************************************/
// @author Alexander Heinecke (Alexander.Heinecke@mytum.de)

#include <x86intrin.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <mpi.h>

/// MPI Stuff
int mpi_err;
// rank of this process
int rank;
// size of global communicator
int size;
//special MPI datatypess
MPI_Datatype row_type;
MPI_Datatype col_type;

// MPI send and receive request objects
MPI_Request allRequests[8];

//and shortcuts to these objects
#define sendLeft   (allRequests[0]) 
#define sendRight  (allRequests[1])
#define sendTop    (allRequests[2])
#define sendBottom (allRequests[3])
#define recvLeft   (allRequests[4])
#define recvRight  (allRequests[5])
#define recvTop    (allRequests[6])
#define recvBottom (allRequests[7])


// coordinates in virtual topology
int topology_size_x, topology_size_y;
// cartesian grid communicator
MPI_Comm cartesian_grid;
// number of points in subgrid
int gridpoints_subgrid_x;
int gridpoints_subgrid_y;
int nb_left, nb_right, nb_top, nb_bottom;
int my_coords[2];


size_t cg_max_iterations;
double cg_eps;

/// store number of grid points in one dimension
std::size_t grid_points_1d = 0;

/// store begin timestep
double begin;


/**
 * initialize and start timer
 */
void timer_start()
{
	begin = MPI_Wtime();
}

/**
 * stop timer and return measured time
 *
 * @return measured time
 */
double timer_stop()
{
	double end = MPI_Wtime();
	double ret = end - begin;

	return ret;
}

/**
 * stores a given grid into a file
 * 
 * MPI COMMUNICATION NEEDED
 * 
 * @param grid the grid that should be stored
 * @param filename the filename
 */
void store_grid(double* grid, std::string filename)
{
	// let processes write to our outputfile one after the other, so we hand a token
	int dummy = 123123;
	int store_tag = 342342431;
	if(rank != 0){
		MPI_Recv(&dummy, 1, MPI_INT, rank-1, store_tag, cartesian_grid, MPI_STATUS_IGNORE);
	}
	//when recv returns, it's out turn
	
	std::fstream filestr;
	if(rank == 0){
		filestr.open (filename.c_str(), std::fstream::out);
	} else {
		filestr.open (filename.c_str(), std::fstream::app | std::fstream::out);
	}
	// calculate mesh width 
	double mesh_width = 1.0/((double)(grid_points_1d-1));

	int xOffset = (gridpoints_subgrid_x-2)*my_coords[0];
	int yOffset = (gridpoints_subgrid_y-2)*my_coords[1];
	// store grid 
	for (int i = 1; i < gridpoints_subgrid_y-1; i++)
	{
		for (int j = 1; j < gridpoints_subgrid_x-1; j++)
		{
			filestr << mesh_width*(i+yOffset) << " " << mesh_width*(j+xOffset) << " " << grid[(i*gridpoints_subgrid_x)+j] << std::endl;
		}
	}
	// store boundary points
	if(nb_left == -1) {
		for(int i = 1; i < gridpoints_subgrid_y-1;i++){
			filestr << mesh_width*(i+yOffset) << " 0 " << grid[(i*gridpoints_subgrid_x)] << std::endl;
		}
	}
	if(nb_right == -1) {
		for(int i = 1; i < gridpoints_subgrid_y-1;i++){
			filestr << mesh_width*(i+yOffset) << " 1 " << grid[((i+1)*gridpoints_subgrid_x)-1] << std::endl;
		}
	}
	if(nb_top == -1) {
		for(int i = 1; i < gridpoints_subgrid_x-1;i++){
			filestr << "0 " << mesh_width*(i+xOffset) << " " << grid[i] << std::endl;
		}
	}
	if(nb_bottom == -1) {
		for(int i = 1; i < gridpoints_subgrid_x-1;i++){
			filestr << "1 " << mesh_width*(i+xOffset) << " " << grid[((gridpoints_subgrid_y-1)*gridpoints_subgrid_x)+i] <<  std::endl;
		}
	}
	//store corner points
	if(nb_left == -1 && nb_top == -1){
		filestr << "0 0 " << grid[0] << std::endl;
	}
	if(nb_top == -1 && nb_right == -1){
		filestr << "0 1 " << grid[gridpoints_subgrid_x-1] << std::endl;
	}
	if(nb_right == -1 && nb_bottom == -1){
		filestr << "1 1 " << grid[gridpoints_subgrid_x*gridpoints_subgrid_y-1] << std::endl;
	}
	if(nb_bottom == -1 && nb_left == -1){
		filestr << "1 0 " << grid[gridpoints_subgrid_x*(gridpoints_subgrid_y-1)] << std::endl;
	}
	

	filestr.close();
	
	//if I'm not the last process, tell the next one that it can write to our file
	if(rank != size-1){
		MPI_Send(&dummy, 1, MPI_INT, rank+1, store_tag, cartesian_grid);
	}
}

/**
 * calculate the grid's initial values for given grid points
 *
 * NO MPI COMMUNICATION
 * 
 * @param x the x-coordinate of a given grid point
 * @param y the y-coordinate of a given grid point
 *
 * @return the initial value at position (x,y)
 */
double eval_init_func(double x, double y)
{
	return (x*x)*(y*y);
}

/**
 * initializes a given grid: inner points are set to zero
 * boundary points are initialized by calling eval_init_func
 *
 * NO MPI COMMUNICATION
 * 
 * @param grid the grid to be initialized
 */
void init_grid(double* grid)
{
	// set all points to zero
	for (int i = 0; i < gridpoints_subgrid_x*gridpoints_subgrid_y; i++)
	{
		grid[i] = 0.0;
	}

	double mesh_width = 1.0/((double)(grid_points_1d-1));
	
	int xOffset = (gridpoints_subgrid_x-2)*my_coords[0];
	int yOffset = (gridpoints_subgrid_y-2)*my_coords[1];
	if(nb_top == -1){
		// along the top corner
		for(int i = 0; i<gridpoints_subgrid_x; i++){
			grid[i]                                                 = eval_init_func((xOffset + i) * mesh_width, 0.0);
		}
	}
	
	if(nb_bottom == -1){
		// along the bottom corner
		for(int i = 0; i<gridpoints_subgrid_x; i++){
			grid[gridpoints_subgrid_x*(gridpoints_subgrid_y-1) + i] = eval_init_func((xOffset + i) * mesh_width, 1.0);
		}
	}
	
	if(nb_left == -1){
		// along left corner
		for(int i = 0; i<gridpoints_subgrid_y; i++){
			grid[gridpoints_subgrid_x*i] = eval_init_func(0.0, (yOffset + i)*mesh_width);
		}
	}
	
	if(nb_right == -1){
		// along right corner
		for(int i = 0; i<gridpoints_subgrid_y; i++){
			grid[(i+1)*gridpoints_subgrid_x-1] = eval_init_func(1.0, (yOffset+i)*mesh_width );
		}
	}
}

/**
 * initializes the right hand side, we want to keep it simple and
 * solve the Laplace equation instead of Poisson (-> b=0)
 * 
 * NO MPI COMMUNICATION 
 *
 * @param b the right hand side
 */
void init_b(double* b)
{
	// set all points to zero
	for (int i = 0; i < gridpoints_subgrid_x*gridpoints_subgrid_y; i++)
	{
		b[i] = 0.0;
	}
}

/**
 * copies data from one grid to another
 *
 * NO MPI COMMUNICATION
 * 
 * @param dest destination grid
 * @param src source grid
 */
void g_copy(double* dest, double* src)
{
	for (int i = 0; i < gridpoints_subgrid_x*gridpoints_subgrid_y; i++)
	{
		dest[i] = src[i];
	}
}

/**
 * calculates the dot product of the two grids (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 * 
 * we only take all inner vertices of our subgrid, otherwise we would sum up sum values twice
 * 
 * MPI COMMUNICATION NEEDED
 *
 * @param grid1 first grid
 * @param grid2 second grid
 */
double g_dot_product(double* grid1, double* grid2)
{
	double tmp = 0.0;

	for (int i = 1; i < gridpoints_subgrid_x-1; i++)
	{
		for (int j = 1; j < gridpoints_subgrid_y-1; j++)
		{
			tmp += (grid1[(i*gridpoints_subgrid_x)+j] * grid2[(i*gridpoints_subgrid_x)+j]);
		}
	}
	double result;
	MPI_Allreduce(&tmp, &result, 1, MPI_DOUBLE, MPI_SUM, cartesian_grid);
	
	return result;
}

/**
 * scales a grid by a given scalar (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 * 
 * NO MPI COMMUNICATION
 * 
 * we scale also our borderpoints, as they are scaled by the other processes as well, so we don't have to communicate these changes
 *
 * @param grid grid to be scaled
 * @param scalar scalar which is used to scale to grid
 */
void g_scale(double* grid, double scalar)
{
	int left, right, top, bottom;
	if(nb_left == -1)
		left = 1;
	else 
		left = 0;
	
	if(nb_right == -1)
		right = gridpoints_subgrid_x - 1;
	else 
		right = gridpoints_subgrid_x;
	
	if(nb_top == -1)
		top = 1;
	else 
		top = 0;
	
	if(nb_bottom == -1)
		bottom = gridpoints_subgrid_y - 1;
	else 
		bottom = gridpoints_subgrid_y;
	
	
	
	for (int i = top; i < bottom; i++)
	{
		for (int j = left; j < right; j++)
		{
			grid[(i*gridpoints_subgrid_x)+j] *= scalar;
		}
	}
}

/**
 * implements BLAS's Xaxpy operation for grids (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 * 
 * NO MPI COMMUNICATION
 *
 * @param dest destination grid
 * @param src source grid
 * @param scalar scalar to scale to source grid
 */
void g_scale_add(double* dest, double* src, double scalar)
{
// 	int left, right, top, bottom;
// 	if(nb_left == -1)
// 		left = 1;
// 	else 
// 		left = 0;
// 	
// 	if(nb_right == -1)
// 		right = gridpoints_subgrid_x - 1;
// 	else 
// 		right = gridpoints_subgrid_x;
// 	
// 	if(nb_top == -1)
// 		top = 1;
// 	else 
// 		top = 0;
// 	
// 	if(nb_bottom == -1)
// 		bottom = gridpoints_subgrid_y - 1;
// 	else 
// 		bottom = gridpoints_subgrid_y;
	
	for (int i = 1; i < gridpoints_subgrid_y-1; i++)
	{
		for (int j = 1; j < gridpoints_subgrid_x-1; j++)
		{
			dest[(i*gridpoints_subgrid_x)+j] += (scalar*src[(i*gridpoints_subgrid_x)+j]);
		}
	}
}

/**
 * implements the the 5-point finite differences stencil (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 * 
 * MPI COMMUNICATION NEEDED
 * 
 * @param grid grid for which the stencil should be evaluated
 * @param result grid where the stencil's evaluation should be stored
 */
void g_product_operator(double* grid, double* result)
{
	double mesh_width = 1.0/((double)(grid_points_1d-1));
	if(nb_left != -1){
		MPI_Isend(&grid[gridpoints_subgrid_x + 1], 1, col_type, nb_left, 0, cartesian_grid, &sendLeft);
		MPI_Irecv(&grid[gridpoints_subgrid_x]    , 1, col_type, nb_left, 0, cartesian_grid, &recvLeft);
	}
	if(nb_right != -1){
		MPI_Isend(&grid[gridpoints_subgrid_x*2-2], 1, col_type, nb_right, 0, cartesian_grid, &sendRight);
		MPI_Irecv(&grid[gridpoints_subgrid_x*2-1], 1, col_type, nb_right, 0, cartesian_grid, &recvRight);
	}
	if(nb_top != -1){
		MPI_Isend(&grid[gridpoints_subgrid_x + 1], 1, row_type, nb_top, 0, cartesian_grid, &sendTop);
		MPI_Irecv(&grid[1]                       , 1, row_type, nb_top, 0, cartesian_grid, &recvTop);
	}
	if(nb_bottom != -1){
		MPI_Isend(&grid[(gridpoints_subgrid_y-2) * gridpoints_subgrid_x + 1], 1, row_type, nb_bottom, 0, cartesian_grid, &sendBottom);
		MPI_Irecv(&grid[(gridpoints_subgrid_y-1) * gridpoints_subgrid_x + 1], 1, row_type, nb_bottom, 0, cartesian_grid, &recvBottom);
	}	
	MPI_Waitall(8, allRequests, MPI_STATUSES_IGNORE);
	
	for (int i = 1; i < gridpoints_subgrid_x-1; i++)
	{
		for (int j = 1; j < gridpoints_subgrid_y-1; j++)
		{
			result[(i*gridpoints_subgrid_x)+j] =  (
							(4.0*grid[(i*gridpoints_subgrid_x)+j]) 
							- grid[((i+1)*gridpoints_subgrid_x)+j]
							- grid[((i-1)*gridpoints_subgrid_x)+j]
							- grid[(i*gridpoints_subgrid_x)+j+1]
							- grid[(i*gridpoints_subgrid_x)+j-1]
							) * (mesh_width*mesh_width);
		}
	}

}

/**
 * The CG Solver (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * For details please see :
 * http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
 *
 * @param grid the grid containing the initial condition
 * @param b the right hand side
 * @param cg_max_iterations max. number of CG iterations 
 * @param cg_eps the CG's epsilon
 */
void solve(double* grid, double* b, std::size_t cg_max_iterations, double cg_eps)
{
	if(rank == 0){
		std::cout << "Starting Conjugated Gradients: (max iterations: " <<cg_max_iterations << ", cg_eps: " << cg_eps << ")" << std::endl;
	}
	
	double eps_squared = cg_eps*cg_eps;
	std::size_t needed_iters = 0;

	// define temporal vectors
	double* q = (double*)_mm_malloc(gridpoints_subgrid_x*gridpoints_subgrid_y*sizeof(double), 64);
	double* r = (double*)_mm_malloc(gridpoints_subgrid_x*gridpoints_subgrid_y*sizeof(double), 64);
	double* d = (double*)_mm_malloc(gridpoints_subgrid_x*gridpoints_subgrid_y*sizeof(double), 64);
	double* b_save = (double*)_mm_malloc(gridpoints_subgrid_x*gridpoints_subgrid_y*sizeof(double), 64);
			
	g_copy(q, grid);
	g_copy(r, grid);
	g_copy(d, grid);
	g_copy(b_save, b);
// 	store_grid(q, "helper_q_parallel.gnuplot");
// 	store_grid(r, "helper_r_parallel.gnuplot");
// 	store_grid(d, "helper_d_parallel.gnuplot");
// 	store_grid(b_save, "helper_b_save_parallel.gnuplot");

	
	double delta_0 = 0.0;
	double delta_old = 0.0;
	double delta_new = 0.0;
	double beta = 0.0;
	double a = 0.0;
	double residuum = 0.0;
	
	g_product_operator(grid, d);
	g_scale_add(b, d, -1.0);
	g_copy(r, b);
	g_copy(d, r);

	// calculate starting norm
	delta_new = g_dot_product(r, r);
	delta_0 = delta_new*eps_squared;
	residuum = (delta_0/eps_squared);
	
	if(rank == 0){
		std::cout << "Starting norm of residuum: " << (delta_0/eps_squared) << std::endl;
		std::cout << "Target norm:               " << (delta_0) << std::endl;
	}
	
	while ((needed_iters < cg_max_iterations) && (delta_new > delta_0))
	{
		// q = A*d
		g_product_operator(d, q);

		// a = d_new / d.q
		a = delta_new/g_dot_product(d, q);
		
		// x = x + a*d
		g_scale_add(grid, d, a);
		
		if ((needed_iters % 50) == 0)
		{
			g_copy(b, b_save);
			g_product_operator(grid, q);
			g_scale_add(b, q, -1.0);
			g_copy(r, b);
		}
		else
		{
			// r = r - a*q
			g_scale_add(r, q, -a);
		}
		
		// calculate new deltas and determine beta
		delta_old = delta_new;
		delta_new = g_dot_product(r, r);
		beta = delta_new/delta_old;

		// adjust d
		g_scale(d, beta);
		g_scale_add(d, r, 1.0);
		
		residuum = delta_new;
		needed_iters++;
		if(rank==0){
			std::cout << "(iter: " << needed_iters << ")delta: " << delta_new << std::endl;
		}
	}

	if(rank == 0) {
		std::cout << "Number of iterations: " << needed_iters << " (max. " << cg_max_iterations << ")" << std::endl;
		std::cout << "Final norm of residuum: " << delta_new << std::endl;
	}
	
	_mm_free(d);
	_mm_free(q);
	_mm_free(r);
	_mm_free(b_save);
}

/**
 * Stores the ranks of all my neighbours (nb_*) as this information does not change during execution
 * 
 * If the variable of a neighbour is -1 the corresponding neighbour does not exist and we therefore are at the border of our domain
 * 
 */
void initNeighbours(){
	int *dims = my_coords;
	
	std::cout << "rank " << rank << "("<< dims[0]<< ", "<< dims[1]<<")";
	// left neighbour
	if(dims[0] == 0){
		nb_left = -1;
	} else {
		int nb_left_dims[] = {dims[0]-1, dims[1]};
		MPI_Cart_rank(cartesian_grid, nb_left_dims, &nb_left);
	}
	
	// right neighbour
	if(dims[0] == topology_size_x - 1){
		nb_right = -1;
	} else {
		int nb_right_dims[] = {dims[0]+1, dims[1]};
		MPI_Cart_rank(cartesian_grid, nb_right_dims, &nb_right);
	}
	
	// top neighbour
	if(dims[1] == 0){
		nb_top = -1;
	} else {
		int nb_top_dims[] = {dims[0], dims[1]-1};
		MPI_Cart_rank(cartesian_grid, nb_top_dims, &nb_top);
	}
	
	// bottom neighbour
	if(dims[1] == topology_size_y - 1){
		nb_bottom = -1;
	} else {
		int nb_bottom_dims[] = {dims[0], dims[1]+1};
		MPI_Cart_rank(cartesian_grid, nb_bottom_dims, &nb_bottom);
	}
	std::cout << "left: " << nb_left << "; right: " << nb_right  << "; top: " << nb_top << "; bottom: " << nb_bottom << std::endl;
}

bool checkAndReadArguments(int argc, char* argv[]){
		// check if all parameters are specified
	if (argc != 6)
	{
		if(rank == 0){
			std::cout << std::endl;
			std::cout << "usage: " << std::endl;
			std::cout << "mpirun -np <num_of_processes> ./app <grid_points_per_dimension> <cg_max_iterations> <cg_eps> <topology_x> <topology_y>" << std::endl;
			std::cout << "make sure that: " << std::endl;
			std::cout << " - <num_of_processes> == <topology_x> * <topology_y>" << std::endl;
			std::cout << " - <grid_points_per_dimension>-2 is dividable by <topology_x> and <topology_y> without remainder" << std::endl;
			std::cout << std::endl;
			std::cout << "example:" << std::endl;
			std::cout << "mpirun -np 16 ./app 9 100 0.0001 4 4 " << std::endl;
			std::cout << std::endl;
		}
		return false;
	}
	

	
	// read cli arguments
	grid_points_1d = atoi(argv[1]);
	cg_max_iterations = atoi(argv[2]);
	cg_eps = atof(argv[3]);
	topology_size_x = atoi(argv[4]);
	topology_size_y = atoi(argv[5]);
	if(size != topology_size_x * topology_size_y){
		if (rank == 0){
			std::cout << "<num_of_processes> != <topology_x> * <topology_y>" << std::endl;
			std::cout << size << " != " << topology_size_x << " * " << topology_size_y << std::endl;
		}
		return false;
	}
	
	if((grid_points_1d-2) % topology_size_x != 0 || (grid_points_1d-2) % topology_size_y != 0){
		if(rank == 0){
			std::cout << " <grid_points_per_dimension>-2 is not dividable by <topology_x> and <topology_y> without remainder" << std::endl;
		}
		return false;
	}
	return true;
}


/**
 * creates special communication topology AND
 * creates special MPI types
 * MPI requests
 */
void setupMPIStuff(){
	int dims[2];
	dims[0] = topology_size_x;
	dims[1] = topology_size_y;
	
	int periods[2];
	periods[0] = 0;
	periods[1] = 0;
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cartesian_grid);
	
	MPI_Comm_rank(cartesian_grid, &rank);
	MPI_Cart_coords(cartesian_grid, rank, 2, my_coords);	
	
	gridpoints_subgrid_x = (grid_points_1d - 2)/topology_size_x + 2;
	gridpoints_subgrid_y = (grid_points_1d - 2)/topology_size_y + 2;
	
	//create new MPI types
	MPI_Type_vector(gridpoints_subgrid_x - 2, 1, 1,                    MPI_DOUBLE, &row_type);
	MPI_Type_vector(gridpoints_subgrid_y - 2, 1, gridpoints_subgrid_x, MPI_DOUBLE, &col_type);
	
	//and commit them
	MPI_Type_commit(&row_type);
	MPI_Type_commit(&col_type);
	
}

/**
 * frees allocated MPI resources (Communicator + Types)
 * 
 */
void tearDownMPIStuff(){
	MPI_Type_free(&col_type);
	MPI_Type_free(&row_type);
	
	MPI_Comm_free(&cartesian_grid);
}

/**
 * main application
 *
 * @param argc number of cli arguments
 * @param argv values of cli arguments
 */
int main(int argc, char* argv[])
{
	// init
	mpi_err = MPI_Init(&argc, &argv);
	if(mpi_err != MPI_SUCCESS){
		printf("Cannot initialize MPI!\n");
		MPI_Finalize();
		exit(-1);
	}
	
	
	// get rank and size
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(!checkAndReadArguments(argc, argv)){
		MPI_Finalize();
		exit(-1);
	}
	setupMPIStuff();
	initNeighbours();

	// initialize the grid and rights hand side
	double* grid = (double*)_mm_malloc(gridpoints_subgrid_x*gridpoints_subgrid_y*sizeof(double), 64);
	double* b = (double*)_mm_malloc(gridpoints_subgrid_x*gridpoints_subgrid_y*sizeof(double), 64);
	init_grid(grid);
	store_grid(grid, "initial_condition_parallel.gnuplot");
	init_b(b);
	store_grid(b, "b.gnuplot");
	
	// solve Poisson equation using CG method
	MPI_Barrier(cartesian_grid);
	if(rank == 0){
		timer_start();
	}
	solve(grid, b, cg_max_iterations, cg_eps);
	MPI_Barrier(cartesian_grid);
	double time;
	if(rank == 0){
		time = timer_stop();
	}
	store_grid(grid, "solution_parallel.gnuplot");
	if(rank == 0){
		std::cout << std::endl << "Needed time: " << time << " s" << std::endl << std::endl;
	}
	
	_mm_free(grid);
	_mm_free(b);

	tearDownMPIStuff();
	
	MPI_Finalize();
	return 0;
}

