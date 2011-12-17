// IMPORTANT NOTE:
// MESH_WIDTH and GRID_POINTS_1D have to be compile time #defines
__kernel void g_product_operator(__global float* grid, __global float* result){
	size_t index_global = get_global_id(1) * GRID_POINTS_1D + get_global_id(0);
	
	result[index_global] =  (4.0*grid[index_global]
							- grid[index_global + 1]
							- grid[index_global - 1]
							- grid[index_global + GRID_POINTS_1D]
							- grid[index_global - GRID_POINTS_1D]
							) * (MESH_WIDTH*MESH_WIDTH);
}

__kernel void g_scale_add(__global float* dest, __global float* src, float scalar)
{
	size_t index = get_global_id(1) * GRID_POINTS_1D + get_global_id(0);
	dest[index] += scalar*src[index];
}

__kernel void g_scale(__global float* grid, float scalar)
{
	size_t index = get_global_id(1) * GRID_POINTS_1D + get_global_id(0);
	grid[index] *= scalar;
}