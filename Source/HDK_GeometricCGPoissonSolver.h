#ifndef HDK_GEOMETRIC_CONJUGATE_GRADIENT_SOLVER_H
#define HDK_GEOMETRIC_CONJUGATE_GRADIENT_SOLVER_H

#include "HDK_GeometricMultigridOperators.h"

namespace HDK
{
    template<typename MatrixVectorMultiplyFunctor,
		typename PreconditionerFunctor,
		typename DotProductFunctor,
		typename L2NormFunctor,
		typename AddScaledVectorFunctor,
		typename StoreReal>
    void solveGeometricConjugateGradient(UT_VoxelArray<StoreReal> &solutionGrid,
					    const UT_VoxelArray<StoreReal> &rhsGrid,
					    MatrixVectorMultiplyFunctor &matrixVectorMultiplyFunctor,
					    PreconditionerFunctor &preconditionerFunctor,
					    DotProductFunctor &dotProductFunctor,
					    L2NormFunctor &l2NormFunctor,
					    AddScaledVectorFunctor &addScaledVectorFunctor,
					    const StoreReal tolerance,
					    const int maxIterations)
    {
	using namespace GeometricMultigridOperators;
	using SolveReal = double;

	UT_Vector3I voxelRes = solutionGrid.getVoxelRes();
	assert(solutionGrid.getVoxelRes() == rhsGrid.getVoxelRes());

	SolveReal rhsNorm = l2NormFunctor(rhsGrid);
	if (rhsNorm == 0)
	{
	    std::cout << "RHS is zero. Nothing to solve" << std::endl;
	    return;
	}

	// Build initial residual vector using an initial guess
	UT_VoxelArray<StoreReal> residualGrid;
	residualGrid.size(voxelRes[0], voxelRes[1], voxelRes[2]);

	matrixVectorMultiplyFunctor(residualGrid, solutionGrid);
	addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);

	SolveReal residualNorm = l2NormFunctor(residualGrid);
    	SolveReal threshold = SolveReal(tolerance) * rhsNorm;

	if (residualNorm < threshold)
	{
	    std::cout << "Residual already below error: " << residualNorm / rhsNorm << std::endl;
	    return;
	}

	// Apply preconditioner for initial search direction
	UT_VoxelArray<StoreReal> pGrid;
	pGrid.size(voxelRes[0], voxelRes[1], voxelRes[2]);
	pGrid.constant(0);

	preconditionerFunctor(pGrid, residualGrid);

	SolveReal rho = dotProductFunctor(pGrid, residualGrid);

	UT_VoxelArray<StoreReal> zGrid;
	zGrid.size(voxelRes[0], voxelRes[1], voxelRes[2]);

	int iteration = 0;
	for (; iteration < maxIterations; ++iteration)
	{
	    // Matrix-vector multiplication
	    matrixVectorMultiplyFunctor(zGrid, pGrid);

	    SolveReal sigma = dotProductFunctor(pGrid, zGrid);
	    SolveReal alpha = rho / sigma;

	    // Update residual
	    addScaledVectorFunctor(residualGrid, residualGrid, zGrid, -alpha);

	    // Update solution
	    addScaledVectorFunctor(solutionGrid, solutionGrid, pGrid, alpha);

	    residualNorm = l2NormFunctor(residualGrid);

	    std::cout << "  Iteration: " << iteration << ". Residual: " << residualNorm << std::endl;

	    if (residualNorm < tolerance)
		break;

	    preconditionerFunctor(zGrid, residualGrid);

	    SolveReal rhoNew = dotProductFunctor(zGrid, residualGrid);
	    SolveReal beta = rhoNew / rho;

	    // Store rho
	    rho = rhoNew;

	    addScaledVectorFunctor(pGrid, zGrid, pGrid, beta);
	}

    	std::cout << "Iterations: " << iteration << std::endl;
	SolveReal error = residualNorm / rhsNorm;
	std::cout << "Drifted relative L2 Error: " << error << std::endl;

	// Recompute residual
	matrixVectorMultiplyFunctor(residualGrid, solutionGrid);
	addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);
	error = l2NormFunctor(residualGrid) / rhsNorm;
	std::cout << "Recomputed relative L2 Error: " << error << std::endl;
    }
}

#endif