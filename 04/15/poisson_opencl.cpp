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
#include <ostream>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include <CL/cl.h>
#include <iosfwd>
#include <sstream>

//adapted from peano project
#define assertionEqualsMsg(lhs,rhs, ...) if ((lhs)!=(rhs)) { \
      std::cout << __VA_ARGS__ << std::endl; \
      exit(-1); \
    }

/// store number of grid points in one dimension
std::size_t grid_points_1d = 0;

/// store begin timestep
struct timeval begin;
/// store end timestep
struct timeval end;

/**
 * initialize and start timer
 */
void timer_start()
{
	gettimeofday(&begin,(struct timezone *)0);
}

cl_platform_id   _clPlatformID;
cl_context       _clContext;
cl_command_queue _clCmdQueue;
cl_program       _clProgram;
cl_kernel        _clKernel;
cl_device_id     _clDeviceID;
cl_int           _clError;

size_t globalOffset[] = {1,1};
size_t globalSize[2];

float* tmp_dot_product_mem1;
float* tmp_dot_product_mem2;


void setupOpenCL(){
	// get ONE available platform
    _clError = clGetPlatformIDs(1, &_clPlatformID, NULL);
    assertionEqualsMsg(_clError, CL_SUCCESS, "Unable to get OpenCL platform.");

    // get ONE OpenCL capable device
    _clError = clGetDeviceIDs(_clPlatformID, CL_DEVICE_TYPE_ALL, 1, &_clDeviceID, NULL);
    assertionEqualsMsg(_clError, CL_SUCCESS, "Unable to get OpenCL device.");

    // create OpenCL context
    cl_context_properties context_properties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)_clPlatformID, 0 };
    _clContext = clCreateContext(context_properties, 1, &_clDeviceID, NULL, NULL, &_clError);
    assertionEqualsMsg(_clError, CL_SUCCESS, "Unable to create OpenCL context.");

    // create OpenCL command queue
    _clCmdQueue = clCreateCommandQueue(_clContext, _clDeviceID, 0, &_clError);
    assertionEqualsMsg(_clError, CL_SUCCESS, "Unable to create OpenCL command queue.");
}

void tearDownOpenCL(){
	clReleaseContext(_clContext);
	clReleaseCommandQueue(_clCmdQueue);
}

void createAndBuildProgram()
{
	std::string kernelProgramSourceFile = "kernelcode.cl";
	std::ifstream kernelSourceFile(kernelProgramSourceFile.c_str());
  assertionEqualsMsg(kernelSourceFile.is_open(), true, "The kerel source file cannot be opened: " << kernelProgramSourceFile
               << std::endl << "Make sure that it exists in the working directory (" << getcwd(NULL, 0) << ")? " << std::endl);
  std::string kernelSource((std::istreambuf_iterator<char>(kernelSourceFile)), std::istreambuf_iterator<char>());
  _clProgram = clCreateProgramWithSource(_clContext, 1, (const char **) &kernelSource, NULL, &_clError);
  assertionEqualsMsg(_clError, CL_SUCCESS, "Error creating programm: " << std::endl << kernelSource)

  std::ostringstream buildOptions;
  buildOptions << "-Werror "
  << " -cl-finite-math-only -cl-strict-aliasing -cl-fast-relaxed-math -cl-single-precision-constant -g"
  << " -DMESH_WIDTH="<<1.0/((float)(grid_points_1d-1))
  << " -DGRID_POINTS_1D=" << grid_points_1d;
 
  char buffer[2048];
  // build programm and wait until the build finished
  _clError = clBuildProgram(_clProgram, 0, NULL, buildOptions.str().c_str(),  NULL, NULL);
  //getBuildInfo
  clGetProgramBuildInfo(_clProgram, _clDeviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
  assertionEqualsMsg(_clError, CL_SUCCESS,
                     "Error building programm ("<< kernelProgramSourceFile <<"): " << std::endl
                     << "buildOptions: " << std::endl << buildOptions.str() << std::endl  << std::endl
                     << "Programm  build log:" <<std::endl << buffer << std::endl);
  printf("Successfully built programm: %s", buffer);
}



/**
 * stop timer and return measured time
 *
 * @return measured time
 */
double timer_stop()
{
	gettimeofday(&end,(struct timezone *)0);
	double seconds, useconds;
	double ret, tmp;

	if (end.tv_usec >= begin.tv_usec)
	{
		seconds = (double)end.tv_sec - (double)begin.tv_sec;
		useconds = (double)end.tv_usec - (double)begin.tv_usec;
	}
	else
	{
		seconds = (double)end.tv_sec - (double)begin.tv_sec;
		seconds -= 1;					// Correction
		useconds = (double)end.tv_usec - (double)begin.tv_usec;
		useconds += 1000000;			// Correction
	}

	// get time in seconds
	tmp = (double)useconds;
	ret = (double)seconds;
	tmp /= 1000000;
	ret += tmp;

	return ret;
}

/**
 * stores a given grid into a file
 * 
 * @param grid the grid that should be stored
 * @param filename the filename
 */
void store_grid(float* grid, std::string filename)
{
	std::fstream filestr;
	filestr.open (filename.c_str(), std::fstream::out);
	
	// calculate mesh width 
	float mesh_width = 1.0/((float)(grid_points_1d-1));

	// store grid incl. boundary points
	for (int i = 0; i < grid_points_1d; i++)
	{
		for (int j = 0; j < grid_points_1d; j++)
		{
			filestr << mesh_width*i << " " << mesh_width*j << " " << grid[(i*grid_points_1d)+j] << std::endl;
		}
		
		filestr << std::endl;
	}

	filestr.close();
}

/**
 * calculate the grid's initial values for given grid points
 *
 * @param x the x-coordinate of a given grid point
 * @param y the y-coordinate of a given grid point
 *
 * @return the initial value at position (x,y)
 */
float eval_init_func(float x, float y)
{
	return (x*x)*(y*y);
}

/**
 * initializes a given grid: inner points are set to zero
 * boundary points are initialized by calling eval_init_func
 *
 * @param grid the grid to be initialized
 */
void init_grid(float* grid)
{
	// set all points to zero
	for (int i = 0; i < grid_points_1d*grid_points_1d; i++)
	{
		grid[i] = 0.0;
	}

	float mesh_width = 1.0/((float)(grid_points_1d-1));
	
	for (int i = 0; i < grid_points_1d; i++)
	{
		// x-boundaries
		grid[i] = eval_init_func(0.0, ((float)i)*mesh_width);
		grid[i + ((grid_points_1d)*(grid_points_1d-1))] = eval_init_func(1.0, ((float)i)*mesh_width);
		// y-boundaries
		grid[i*grid_points_1d] = eval_init_func(((float)i)*mesh_width, 0.0);
		grid[(i*grid_points_1d) + (grid_points_1d-1)] = eval_init_func(((float)i)*mesh_width, 1.0);
	}
}

/**
 * initializes the right hand side, we want to keep it simple and
 * solve the Laplace equation instead of Poisson (-> b=0)
 *
 * @param b the right hand side
 */
void init_b(float* b)
{
	// set all points to zero
	for (int i = 0; i < grid_points_1d*grid_points_1d; i++)
	{
		b[i] = 0.0;
	}
}

/**
 * copies data from one grid on device to another on device
 *
 * @param dest destination grid
 * @param src source grid
 */
void g_copy_dev_dev(cl_mem dest, cl_mem src)
{
	_clError = clEnqueueCopyBuffer(_clCmdQueue, src, dest, 0, 0, grid_points_1d*grid_points_1d*sizeof(float), 0, NULL, NULL);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error copying from src(device) to dest(device).");
}

/**
 * copies data from one grid on device to host
 *
 * @param dest destination grid
 * @param src source grid
 */
void g_copy_dev_host(float* dest, cl_mem src)
{
	_clError = clEnqueueReadBuffer(_clCmdQueue, src, CL_TRUE, 0, grid_points_1d*grid_points_1d*sizeof(float), dest, 0, NULL, NULL);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error copying from src(device) to dest(host).");
}


/**
 * copies data from one grid on host to device
 *
 * @param dest destination grid
 * @param src source grid
 */
void g_copy_host_dev(cl_mem dest, float* src)
{
	_clError = clEnqueueWriteBuffer(_clCmdQueue, dest, CL_TRUE, 0, grid_points_1d*grid_points_1d*sizeof(float), src, 0, NULL, NULL);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error copying from src (host) to dest (device).");
}


/**
 * calculates the dot product of the two grids (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param grid1 first grid
 * @param grid2 second grid
 */
float g_dot_product(cl_mem clgrid1, cl_mem clgrid2)
{
	float tmp = 0.0;
	float* grid1 = tmp_dot_product_mem1;
	float* grid2 = tmp_dot_product_mem2;
	
	g_copy_dev_host(grid1, clgrid1);
	if(clgrid1 != clgrid2){
		g_copy_dev_host(grid2, clgrid2);
	} else {
		grid2 = grid1;
	}
	
	for (int i = 1; i < grid_points_1d-1; i++)
	{
		for (int j = 1; j < grid_points_1d-1; j++)
		{
			tmp += (grid1[(i*grid_points_1d)+j] * grid2[(i*grid_points_1d)+j]);
		}
	}
	
	return tmp;
}

/**
 * scales a grid by a given scalar (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param grid grid to be scaled
 * @param scalar scalar which is used to scale to grid
 */
void g_scale(cl_mem grid, float scalar)
{
	cl_kernel g_scale_kernel = clCreateKernel(_clProgram, "g_scale", &_clError);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error creating kernel: g_scale_kernel.");

	_clError = clSetKernelArg(g_scale_kernel, 0, sizeof(cl_mem), &grid);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error setting kernel arg: grid.");
	
	_clError = clSetKernelArg(g_scale_kernel, 1, sizeof(float), &scalar);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error setting kernel arg: scalar.");

	_clError = clEnqueueNDRangeKernel(_clCmdQueue, g_scale_kernel, 2, globalOffset, globalSize, NULL, 0, NULL, NULL);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error enqueuing kernel: g_scale_kernel.");

	clReleaseKernel(g_scale_kernel);
}

/**
 * implements BLAS's Xaxpy operation for grids (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 *
 * @param dest destination grid
 * @param src source grid
 * @param scalar scalar to scale to source grid
 */
void g_scale_add(cl_mem dest, cl_mem src, float scalar)
{
	cl_kernel g_scale_add_kernel = clCreateKernel(_clProgram, "g_scale_add", &_clError);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error creating kernel: g_scale_add_kernel.");

	_clError = clSetKernelArg(g_scale_add_kernel, 0, sizeof(cl_mem),& dest);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error setting kernel arg: dest.");
	
	_clError = clSetKernelArg(g_scale_add_kernel, 1, sizeof(cl_mem), &src);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error setting kernel arg: src.");

	_clError = clSetKernelArg(g_scale_add_kernel, 2, sizeof(float), &scalar);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error setting kernel arg: scalar.");

	_clError = clEnqueueNDRangeKernel(_clCmdQueue, g_scale_add_kernel, 2, globalOffset, globalSize, NULL, 0, NULL, NULL);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error enqueuing kernel: g_scale_add_kernel.");

	clReleaseKernel(g_scale_add_kernel);
}

/**
 * implements the the 5-point finite differences stencil (only inner grid points are modified due 
 * to Dirichlet boundary conditions)
 * 
 * @param grid grid for which the stencil should be evaluated
 * @param result grid where the stencil's evaluation should be stored
 */
void g_product_operator(cl_mem grid, cl_mem result)
{
	cl_kernel g_product_operator_kernel = clCreateKernel(_clProgram, "g_product_operator", &_clError);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error creating kernel: g_product_operator.");

	_clError = clSetKernelArg(g_product_operator_kernel, 0, sizeof(cl_mem), &grid);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error setting kernel arg: grid.");
	
	_clError = clSetKernelArg(g_product_operator_kernel, 1, sizeof(cl_mem), &result);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error setting kernel arg: result.");

	_clError = clEnqueueNDRangeKernel(_clCmdQueue, g_product_operator_kernel, 2, globalOffset, globalSize, NULL, 0, NULL, NULL);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error enqueuing kernel: g_product_operator_kernel.");

	clReleaseKernel(g_product_operator_kernel);
}

cl_mem allocateDeviceBuffer(){
	size_t gridsize = grid_points_1d*grid_points_1d*sizeof(float);
	cl_mem result = clCreateBuffer(_clContext, CL_MEM_READ_WRITE, gridsize, NULL, &_clError);
	assertionEqualsMsg(_clError, CL_SUCCESS, "Error allocating device memory of size " << gridsize << ". Error: " << _clError);
	return result;
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
void solve(float* grid, float* b, std::size_t cg_max_iterations, float cg_eps)
{
	std::cout << "Starting Conjugated Gradients" << std::endl;

	float eps_squared = cg_eps*cg_eps;
	std::size_t needed_iters = 0;

	setupOpenCL();
	createAndBuildProgram();
	
	// define temporal vectors, these are allocated on the device
	cl_mem _cl_b = allocateDeviceBuffer();
	cl_mem _cl_grid = allocateDeviceBuffer();
	cl_mem _cl_q = allocateDeviceBuffer();
	cl_mem _cl_r = allocateDeviceBuffer();
	cl_mem _cl_d = allocateDeviceBuffer();
	
	tmp_dot_product_mem1 = (float*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(float), 64);
	tmp_dot_product_mem2 = (float*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(float), 64);

	
	g_copy_host_dev(_cl_grid, grid);
	g_copy_host_dev(_cl_b, b);
	
	g_copy_dev_dev(_cl_q, _cl_grid);
	g_copy_dev_dev(_cl_r, _cl_grid);
	g_copy_dev_dev(_cl_d, _cl_grid);
	
	float delta_0 = 0.0;
	float delta_old = 0.0;
	float delta_new = 0.0;
	float beta = 0.0;
	float a = 0.0;
	float residuum = 0.0;
	
	g_product_operator(_cl_grid, _cl_d);
	g_scale_add(_cl_b, _cl_d, -1.0);
	g_copy_dev_dev(_cl_r,  _cl_b);
	g_copy_dev_dev(_cl_d, _cl_r);

	// calculate starting norm
	delta_new = g_dot_product(_cl_r, _cl_r); //TODO implement for opencl
	delta_0 = delta_new*eps_squared;
	residuum = (delta_0/eps_squared);
	
	std::cout << "Starting norm of residuum: " << (delta_0/eps_squared) << std::endl;
	std::cout << "Target norm:               " << (delta_0) << std::endl;

	while ((needed_iters < cg_max_iterations) && (delta_new > delta_0))
	{
		// q = A*d
		g_product_operator(_cl_d, _cl_q);

		// a = d_new / d.q
		a = delta_new/g_dot_product(_cl_d, _cl_q);
		
		// x = x + a*d
		g_scale_add(_cl_grid, _cl_d, a);
		
		if ((needed_iters % 50) == 0)
		{
			g_copy_host_dev(_cl_b, b);
			g_product_operator(_cl_grid, _cl_q);
			g_scale_add(_cl_b, _cl_q, -1.0);
			g_copy_dev_dev(_cl_r, _cl_b);
		}
		else
		{
			// r = r - a*q
			g_scale_add(_cl_r, _cl_q, -a);
		}
		
		// calculate new deltas and determine beta
		delta_old = delta_new;
		delta_new = g_dot_product(_cl_r, _cl_r);
		beta = delta_new/delta_old;

		// adjust d
		g_scale(_cl_d, beta);
		g_scale_add(_cl_d, _cl_r, 1.0);
		
		residuum = delta_new;
		needed_iters++;
		std::cout << "(iter: " << needed_iters << ")delta: " << delta_new << std::endl;
	}
	g_copy_dev_host(grid, _cl_grid);

	std::cout << "Number of iterations: " << needed_iters << " (max. " << cg_max_iterations << ")" << std::endl;
	std::cout << "Final norm of residuum: " << delta_new << std::endl;
	
	clReleaseMemObject(_cl_b);
	clReleaseMemObject(_cl_grid);
	clReleaseMemObject(_cl_q);
	clReleaseMemObject(_cl_r);
	clReleaseMemObject(_cl_d);
	
	_mm_free(tmp_dot_product_mem1);
	_mm_free(tmp_dot_product_mem2);
	tearDownOpenCL();
}

/**
 * main application
 *
 * @param argc number of cli arguments
 * @param argv values of cli arguments
 */
int main(int argc, char* argv[])
{
	// check if all parameters are specified
	if (argc != 4)
	{
		std::cout << std::endl;
		std::cout << "meshwidth" << std::endl;
		std::cout << "cg_max_iterations" << std::endl;
		std::cout << "cg_eps" << std::endl;
		std::cout << std::endl;
		std::cout << "example:" << std::endl;
		std::cout << "./app 0.125 100 0.0001" << std::endl;
		std::cout << std::endl;
		
		return -1;
	}
	
	// read cli arguments
	float mesh_width = atof(argv[1]);
	size_t cg_max_iterations = atoi(argv[2]);
	float cg_eps = atof(argv[3]);

	// calculate grid points per dimension
	grid_points_1d = (std::size_t)(1.0/mesh_width)+1;
	globalSize[0] = globalSize[1] = grid_points_1d-2;
	
	// initialize the gird and rights hand side
	float* grid = (float*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(float), 64);
	float* b = (float*)_mm_malloc(grid_points_1d*grid_points_1d*sizeof(float), 64);
	
	init_grid(grid);
	store_grid(grid, "initial_condition.gnuplot");
	init_b(b);
	store_grid(b, "b.gnuplot");
	
	// solve Poisson equation using CG method
	timer_start();
	solve(grid, b, cg_max_iterations, cg_eps);
	double time = timer_stop();
	store_grid(grid, "solution.gnuplot");
	
	std::cout << std::endl << "Needed time: " << time << " s" << std::endl << std::endl;
	
	_mm_free(grid);
	_mm_free(b);

	return 0;
}

