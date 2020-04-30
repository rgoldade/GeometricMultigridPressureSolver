# GeometricMultigridPressureSolver

This is a geometric multigrid solver for a standard scalar poisson problem that is typically used to solve for pressure in fluids simulations. It is based on McAdams et al., 2010 and built as a plug-in to Houdini 18.

## Build

To build this project in Houdini (Linux):

1. Install Houdini 18.0 or higher.

2. Go to install folder (/opt/hfs.xx).

3. Type "source houdini_setup" to get the necessary environment variables.

4. Make a build folder the top level of the repository.

5. Run cmake .. in the build folder (if Eigen is preferred over native Houdini's Conjugate Gradient method, include -DUSEEIGEN=ON).

6. Run make in the build folder.

7. Verify that it was added to Houdini by:
  - Launch Houdini.
  - Press "tab" in the Network Editor and select a "DOP Network" and place it in the editor.
  - Jump into the DOP Network, press "tab" again and verify that "HDK Geometric Free Surface Pressure Solver" is searchable.

To build this project in another OS, please refer to the [Houdini HDK](https://www.sidefx.com/docs/hdk/).

## Scenes

The Scenes folder contains two example files. *testMultiGrid* is simply a diagnostic tool to verify solver convergence and symmetry. *flipSplash* is an example simulation where liquid objects splash into a pool of liquid. Both the standard free surface pressure solver and the geometric multigrid accelerated pressure solver have been added into the FLIP Solver node and should be immediately usable after successfully compiling the code from Source.

## TO-DO
1. Introduce mixed precision according to Narrow-Band Topology Optimization on a Sparsely Populated Grid [Liu et al., 2018]

