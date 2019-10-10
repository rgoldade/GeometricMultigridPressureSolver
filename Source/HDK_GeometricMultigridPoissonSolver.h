#ifndef HDK_GEOMETRIC_MULTIGRID_POISSON_SOLVER_H
#define HDK_GEOMETRIC_MULTIGRID_POISSON_SOLVER_H

#include <Eigen/Sparse>

#include <UT/UT_VoxelArray.h>

namespace HDK
{
    class GeometricMultigridPoissonSolver
    {
	static constexpr int UNLABELLED_CELL = -1;

	using StoreReal = float;
	using SolveReal = double;
	using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

    public:

	GeometricMultigridPoissonSolver(const UT_VoxelArray<int> &initialDomainCellLabels,
					const int mgLevels,
					const SolveReal dx,
					const int boundarySmootherWidth,
					const int boundarySmootherIterations,
					const bool useGaussSeidel);

	void setBoundaryWeights(const std::array<UT_VoxelArray<StoreReal>, 3>  &boundaryWeights);

	void applyVCycle(UT_VoxelArray<StoreReal> &solutionVector,
			    const UT_VoxelArray<StoreReal> &rhsVector,
			    const bool useInitialGuess = false);

    private:

	UT_Array<UT_VoxelArray<int>> myDomainCellLabels;
	UT_Array<UT_VoxelArray<StoreReal>> mySolutionGrids, myRHSGrids, myResidualGrids;

	UT_Array<UT_Array<UT_Vector3I>> myBoundaryCells;

	UT_VoxelArray<exint> myDirectSolverIndices;

	int myMGLevels;
	UT_Vector3I myExteriorOffset;

	bool myDoApplyBoundaryWeights;
	std::array<UT_VoxelArray<StoreReal>, 3> myFineBoundaryWeights;

	UT_Array<SolveReal> myDx;

	const int myBoundarySmootherWidth;
	const int myBoundarySmootherIterations;

	const bool myUseGaussSeidel;

	Vector myCoarseRHSVector;
	Eigen::SparseMatrix<SolveReal> sparseMatrix;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolveReal>> myCoarseSolver;
    };
}
#endif