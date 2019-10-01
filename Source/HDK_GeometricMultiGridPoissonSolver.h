#ifndef HDK_GEOMETRIC_MULTI_GRID_POISSON_SOLVER_H
#define HDK_GEOMETRIC_MULTI_GRID_POISSON_SOLVER_H

#include "Eigen/Sparse"

#include <UT/UT_VoxelArray.h>

namespace HDK
{
    class GeometricMultiGridPoissonSolver
    {
	static constexpr int UNLABELLED_CELL = -1;

	using StoreReal = float;
	using SolveReal = double;
	using Vector = Eigen::VectorXd;

    public:

	GeometricMultiGridPoissonSolver(const UT_VoxelArray<int> &initialDomainCellLabels,
					const int mgLevels,
					const SolveReal dx,
					const int boundarySmootherWidth = 2,
					const int boundarySmootherIterations = 1,
					const int smootherIterations = 1,
					const bool useGaussSeidel = false);

	void setGradientWeights(const UT_VoxelArray<StoreReal> (&gradientWeights)[3]);

	void applyVCycle(UT_VoxelArray<StoreReal> &solutionVector,
			    UT_VoxelArray<StoreReal> &rhsVector,
			    const bool useInitialGuess = false);

    private:

	UT_Array<UT_VoxelArray<int>> myDomainCellLabels;
	UT_Array<UT_VoxelArray<StoreReal>> mySolutionGrids, myRHSGrids, myResidualGrids;

	UT_Array<UT_Array<UT_Vector3I>> myBoundaryCells;

	UT_VoxelArray<exint> myDirectSolverIndices;

	int myMGLevels;
	UT_Vector3I myExteriorOffset;

	bool myDoApplyGradientWeights;
	UT_VoxelArray<StoreReal> myFineGradientWeights[3];

	UT_Array<SolveReal> myDx;

	const int myBoundarySmootherWidth;
	const int myBoundarySmootherIterations;
	const int myTotalSmootherIterations;

	const bool myUseGaussSeidel;

	Vector myCoarseRHSVector;
	Eigen::SparseMatrix<SolveReal> sparseMatrix;
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolveReal>> myCoarseSolver;
    };
}
#endif