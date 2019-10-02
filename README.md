# GeometricMultiGridPressureSolver

This is a geometric multigrid solver for a standard scalar poisson problem that is typically used to solve for pressure in fluids simulations. It is based on McAdams et al., 2010 and built as a plug-in to Houdini 18.

## TO-DO
1. Optimize downstroke of the multigrid v-cycle when using a zero initial guess (as is the case when using MG as a preconditioner).
2. Factor out dx terms
3. Remove weights term from interior smoother (only needed at the boundary for ghost fluid and cut-cells)
