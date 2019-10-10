# GeometricMultigridPressureSolver

This is a geometric multigrid solver for a standard scalar poisson problem that is typically used to solve for pressure in fluids simulations. It is based on McAdams et al., 2010 and built as a plug-in to Houdini 18.

## TO-DO
1. Factor out dx terms
2. Optimize MG build operations
3. Handle voxel probe rotations like in UT_VoxelProbeCube in Houdini's HDK
4. Write a free surface pressure solver plug-in utilizing the geometric MG preconditioned CG solver
