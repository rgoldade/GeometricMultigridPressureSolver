#ifndef HDK_GEOMETRIC_CONJUGATE_GRADIENT_SOLVER_H
#define HDK_GEOMETRIC_CONJUGATE_GRADIENT_SOLVER_H

#include <UT/UT_StopWatch.h>

#include "HDK_GeometricMultigridOperators.h"


namespace HDK
{
    template<typename MatrixVectorMultiplyFunctor,
		typename PreconditionerFunctor,
		typename DotProductFunctor,
		typename SquaredL2NormFunctor,
		typename AddToVectorFunctor,
		typename AddScaledVectorFunctor,
		typename StoreReal>
    void solveGeometricConjugateGradient(UT_VoxelArray<StoreReal> &solutionGrid,
					    const UT_VoxelArray<StoreReal> &rhsGrid,
					    MatrixVectorMultiplyFunctor &matrixVectorMultiplyFunctor,
					    PreconditionerFunctor &preconditionerFunctor,
					    DotProductFunctor &dotProductFunctor,
					    SquaredL2NormFunctor &squaredNormFunctor,
					    AddToVectorFunctor &addToVectorFunctor,
					    AddScaledVectorFunctor &addScaledVectorFunctor,
					    const StoreReal tolerance,
					    const int maxIterations)
    {
	using namespace GeometricMultigridOperators;
	using SolveReal = double;

	UT_Vector3I voxelRes = solutionGrid.getVoxelRes();
	assert(solutionGrid.getVoxelRes() == rhsGrid.getVoxelRes());

	SolveReal rhsNorm2 = squaredNormFunctor(rhsGrid);
	if (rhsNorm2 == 0)
	{
	    std::cout << "RHS is zero. Nothing to solve" << std::endl;
	    return;
	}

	// Build initial residual vector using an initial guess
	UT_VoxelArray<StoreReal> residualGrid;
	residualGrid.size(voxelRes[0], voxelRes[1], voxelRes[2]);

	{
	    UT_StopWatch timer;
	    timer.start();

	    matrixVectorMultiplyFunctor(residualGrid, solutionGrid);
	    addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);

	    auto time = timer.stop();
	    std::cout << "    Compute initial residual time: " << time << std::endl;
	}

	SolveReal residualNorm2 = squaredNormFunctor(residualGrid);
    	SolveReal threshold = SolveReal(tolerance) * SolveReal(tolerance) * rhsNorm2;

	if (residualNorm2 < threshold)
	{
	    std::cout << "Residual already below error: " << SYSsqrt(residualNorm2 / rhsNorm2) << std::endl;
	    return;
	}

	// Apply preconditioner for initial search direction
	UT_VoxelArray<StoreReal> pGrid;
	pGrid.size(voxelRes[0], voxelRes[1], voxelRes[2]);
	pGrid.constant(0);

	{
	    UT_StopWatch timer;
	    timer.start();
	    
	    preconditionerFunctor(pGrid, residualGrid);

	    auto time = timer.stop();
	    std::cout << "    Apply initial preconditioner time: " << time << std::endl;
	}
	
	SolveReal absNew;
	{
	    UT_StopWatch timer;
	    timer.start();
	    
	    absNew = dotProductFunctor(pGrid, residualGrid);

	    auto time = timer.stop();
	    std::cout << "    Initial dot product time: " << time << std::endl;
	}

	UT_VoxelArray<StoreReal> zGrid;
	zGrid.size(voxelRes[0], voxelRes[1], voxelRes[2]);
	zGrid.constant(0);

	UT_VoxelArray<StoreReal> tempGrid;
	tempGrid.size(voxelRes[0], voxelRes[1], voxelRes[2]);
	tempGrid.constant(0);

	int iteration = 0;
	for (; iteration < maxIterations; ++iteration)
	{
	    std::cout << "  Iteration: " << iteration << std::endl;

	    {
		UT_StopWatch timer;
		timer.start();

		// Matrix-vector multiplication
		matrixVectorMultiplyFunctor(tempGrid, pGrid);

		auto time = timer.stop();
		std::cout << "    Matrix-vector multiply time: " << time << std::endl;
	    }

	    SolveReal alpha;
	    {
		UT_StopWatch timer;
		timer.start();
	    
		alpha = absNew / dotProductFunctor(pGrid, tempGrid);

		auto time = timer.stop();
		std::cout << "    Alpha dot product time: " << time << std::endl;
	    }

	    {
		UT_StopWatch timer;
		timer.start();
	    
		// Update solution
		addToVectorFunctor(solutionGrid, pGrid, alpha);

		auto time = timer.stop();
		std::cout << "    Update solution time: " << time << std::endl;
	    }	

	    {
		UT_StopWatch timer;
		timer.start();
	    
		// Update residual
		addToVectorFunctor(residualGrid, tempGrid, -alpha);

		auto time = timer.stop();
		std::cout << "    Update residual time: " << time << std::endl;
	    }

	    {
		UT_StopWatch timer;
		timer.start();
	    
		residualNorm2 = squaredNormFunctor(residualGrid);
		
		auto time = timer.stop();
		std::cout << "    L-2 norm time: " << time << std::endl;
	    }	

	    std::cout << "    Relative error: " << SYSsqrt(residualNorm2 / rhsNorm2) << std::endl;

	    if (residualNorm2 < threshold)
		break;

	    {
		UT_StopWatch timer;
		timer.start();
	    
		preconditionerFunctor(zGrid, residualGrid);
		
		auto time = timer.stop();
		std::cout << "    Apply preconditioner time: " << time << std::endl;
	    }	
	    
	    SolveReal absOld = absNew;
	    SolveReal beta;
	    {
		UT_StopWatch timer;
		timer.start();
	    
		absNew = dotProductFunctor(zGrid, residualGrid);
		beta = absNew / absOld;
		
		auto time = timer.stop();
		std::cout << "    Apply dot product: " << time << std::endl;
	    }

	    {
		UT_StopWatch timer;
		timer.start();
	    
		addScaledVectorFunctor(pGrid, zGrid, pGrid, beta);
		
		auto time = timer.stop();
		std::cout << "    Update search direction: " << time << std::endl;
	    }
	}

    	std::cout << "Iterations: " << iteration << std::endl;
	SolveReal error = SYSsqrt(residualNorm2 / rhsNorm2);
	std::cout << "Drifted relative L2 Error: " << error << std::endl;

	// Recompute residual
	matrixVectorMultiplyFunctor(residualGrid, solutionGrid);
	addScaledVectorFunctor(residualGrid, rhsGrid, residualGrid, -1);
	error = SYSsqrt(squaredNormFunctor(residualGrid) / rhsNorm2);
	std::cout << "Recomputed relative L2 Error: " << error << std::endl;
    }
}

#endif