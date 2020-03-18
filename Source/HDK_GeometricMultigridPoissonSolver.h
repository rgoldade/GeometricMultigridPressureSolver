#ifndef HDK_GEOMETRIC_MULTIGRID_POISSON_SOLVER_H
#define HDK_GEOMETRIC_MULTIGRID_POISSON_SOLVER_H

#include <Eigen/Sparse>

#include <UT/UT_VoxelArray.h>

namespace HDK
{
    class GeometricMultigridPoissonSolver
    {
	static constexpr int UNLABELLED_CELL = -1;

	using StoreReal = double;
	using SolveReal = double;
	using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

    public:

	GeometricMultigridPoissonSolver(const UT_VoxelArray<int> &initialCellLabels,
					const std::array<UT_VoxelArray<StoreReal>, 3>  &boundaryWeights,
					const int mgLevels,
					const bool useGaussSeidel,
					const bool doPrintStats = false);

	void
	applyVCycle(UT_VoxelArray<StoreReal> &solutionVector,
		    const UT_VoxelArray<StoreReal> &rhsVector,
		    const bool useInitialGuess = false);

	int getMGLevels() { return myMGLevels; }

    private:

	UT_Array<UT_VoxelArray<int>> myCellLabels;
	UT_Array<UT_VoxelArray<StoreReal>> mySolutionGrids, myRHSGrids, myResidualGrids;

	UT_Array<UT_Array<UT_Vector3I>> myBoundaryCells;

	int myMGLevels;

	std::array<UT_VoxelArray<StoreReal>, 3> myFineBoundaryWeights;

	const int myBoundarySmootherWidth;
	const int myBoundarySmootherIterations;

	const bool myUseGaussSeidel;
	const bool myDoPrintStats;

	UT_VoxelArray<int> myDirectSolverIndices;
	Eigen::SparseMatrix<SolveReal> mySparseMatrix;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolveReal>> myCoarseSolver;
    };
}
#endif