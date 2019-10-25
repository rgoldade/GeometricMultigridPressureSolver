#include "HDK_GeometricMultigridPoissonSolver.h"

#include <SIM/SIM_FieldUtils.h>
#include <UT/UT_ParallelUtil.h>

#include <UT/UT_StopWatch.h>

#include "HDK_GeometricMultigridOperators.h"

namespace HDK
{
    template<typename Vector, typename StoreReal>
    void
    copyGridToVector(Vector &vector,
			const UT_VoxelArray<StoreReal> &gridVector,
			const UT_VoxelArray<exint> &gridIndices,
			const UT_VoxelArray<int> &cellLabels)
    {
	using namespace HDK::GeometricMultigridOperators;

	UT_Interrupt *boss = UTgetInterrupt();

	assert(gridVector.getVoxelRes() == gridIndices.getVoxelRes() &&
		gridIndices.getVoxelRes() == cellLabels.getVoxelRes());

	UTparallelForEachNumber(gridIndices.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<exint> vit;
	    vit.setConstArray(&gridIndices);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> gridVectorProbe;
	    gridVectorProbe.setConstArray(&gridVector);

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.atEnd())
		{
		    if (!vit.isTileConstant())
		    {
			for (; !vit.atEnd(); vit.advance())
			{
			    exint index = vit.getValue();

			    if (index >= 0)
			    {
#if !defined(NDEBUG)
				UT_Vector3I cell(vit.x(), vit.y(), vit.z());
				assert(cellLabels(cell) == CellLabels::INTERIOR_CELL ||
					cellLabels(cell) == CellLabels::BOUNDARY_CELL);
#endif
				gridVectorProbe.setIndex(vit);
				vector(index) = gridVectorProbe.getValue();
			    }
#if !defined(NDEBUG)
			    else
			    {
				UT_Vector3I cell(vit.x(), vit.y(), vit.z());
				assert(!(cellLabels(cell) == CellLabels::INTERIOR_CELL ||
					cellLabels(cell) == CellLabels::BOUNDARY_CELL));
			    }
#endif
			}
		    }
		}
	    }
	});
    }
    
    template<typename Vector, typename StoreReal>
    void
    copyVectorToGrid(UT_VoxelArray<StoreReal> &gridVector,
			const Vector &vector,
			const UT_VoxelArray<exint> &gridIndices,
			const UT_VoxelArray<int> &cellLabels)
    {
	using namespace HDK::GeometricMultigridOperators;

	UT_Interrupt *boss = UTgetInterrupt();

	assert(gridVector.getVoxelRes() == gridIndices.getVoxelRes() &&
		gridIndices.getVoxelRes() == cellLabels.getVoxelRes());

	UTparallelForEachNumber(gridIndices.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<exint> vit;
	    vit.setConstArray(&gridIndices);

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> gridVectorProbe;
	    gridVectorProbe.setArray(&gridVector);

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.atEnd())
		{
		    if (!vit.isTileConstant())
		    {
			for (; !vit.atEnd(); vit.advance())
			{
			    exint index = vit.getValue();

			    if (index >= 0)
			    {
#if !defined(NDEBUG)
				UT_Vector3I cell(vit.x(), vit.y(), vit.z());
				assert(cellLabels(cell) == CellLabels::INTERIOR_CELL ||
					cellLabels(cell) == CellLabels::BOUNDARY_CELL);
#endif				
				gridVectorProbe.setIndex(vit);
				gridVectorProbe.setValue(vector(index));
			    }
#if !defined(NDEBUG)
			    else
			    {
				UT_Vector3I cell(vit.x(), vit.y(), vit.z());
				assert(!(cellLabels(cell) == CellLabels::INTERIOR_CELL ||
					cellLabels(cell) == CellLabels::BOUNDARY_CELL));
			    }
#endif
			}
		    }
		}
	    }
	});

    }

    GeometricMultigridPoissonSolver::GeometricMultigridPoissonSolver(const UT_VoxelArray<int> &initialDomainCellLabels,
									const std::array<UT_VoxelArray<StoreReal>, 3>  &boundaryWeights,
									const int mgLevels,
									const SolveReal dx,
									const bool useGaussSeidel)
    : myMGLevels(mgLevels)
    , myBoundarySmootherWidth(3)
    , myBoundarySmootherIterations(3)
    , myUseGaussSeidel(useGaussSeidel)
    {
	using namespace HDK::GeometricMultigridOperators;
	using namespace SIM::FieldUtils;

	UT_StopWatch timer;
	timer.start();

	assert(myMGLevels > 0);
	assert(dx > 0);

	// Verify that the initial domain cells are properly sized and contain the necessary EXTERIOR padding
	assert(initialDomainCellLabels.getVoxelRes()[0] % 2 == 0 &&
		initialDomainCellLabels.getVoxelRes()[1] % 2 == 0 &&
		initialDomainCellLabels.getVoxelRes()[2] % 2 == 0);

	assert(int(std::log2(initialDomainCellLabels.getVoxelRes()[0])) + 1 >= mgLevels &&
		int(std::log2(initialDomainCellLabels.getVoxelRes()[1])) + 1 >= mgLevels &&
		int(std::log2(initialDomainCellLabels.getVoxelRes()[2])) + 1 >= mgLevels);

	myDomainCellLabels.setSize(myMGLevels);
	myDomainCellLabels[0] = initialDomainCellLabels;
	myDomainCellLabels[0].collapseAllTiles();

	assert( boundaryWeights[0].getVoxelRes()[0] - 1 == myDomainCellLabels[0].getVoxelRes()[0] &&
		boundaryWeights[0].getVoxelRes()[1]     == myDomainCellLabels[0].getVoxelRes()[1] &&
		boundaryWeights[0].getVoxelRes()[2]     == myDomainCellLabels[0].getVoxelRes()[2] &&

		boundaryWeights[1].getVoxelRes()[0]     == myDomainCellLabels[0].getVoxelRes()[0] &&
		boundaryWeights[1].getVoxelRes()[1] - 1 == myDomainCellLabels[0].getVoxelRes()[1] &&
		boundaryWeights[1].getVoxelRes()[2]     == myDomainCellLabels[0].getVoxelRes()[2] &&

		boundaryWeights[2].getVoxelRes()[0]     == myDomainCellLabels[0].getVoxelRes()[0] &&
		boundaryWeights[2].getVoxelRes()[1]     == myDomainCellLabels[0].getVoxelRes()[1] &&
		boundaryWeights[2].getVoxelRes()[2] - 1 == myDomainCellLabels[0].getVoxelRes()[2]);

	for (int axis : {0,1,2})
	    myFineBoundaryWeights[axis] = boundaryWeights[axis];

	auto time = timer.stop();
	std::cout << "      Copy initial domain time: " << time << std::endl;
	timer.clear();
	timer.start();

	auto checkSolvableCell = [&](const UT_VoxelArray<int> &testGrid) -> bool
	{
	    bool hasSolvableCell = false;
	    
	    UT_Interrupt *boss = UTgetInterrupt();

	    UTparallelForEachNumber(testGrid.numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		if (hasSolvableCell) return;

		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&testGrid);

		for (int i = range.begin(); i != range.end(); ++i)
		{
		    vit.myTileStart = i;
		    vit.myTileEnd = i + 1;
		    vit.rewind();

		    if (boss->opInterrupt())
			break;

		    if (!vit.atEnd())
		    {
			if (!vit.isTileConstant())
			{
			    for (; !vit.atEnd(); vit.advance())
			    {
				if (vit.getValue() == CellLabels::INTERIOR_CELL ||
				    vit.getValue() == CellLabels::BOUNDARY_CELL)
				{
				    hasSolvableCell = true;
				    return;
				}
			    }
			}
			else if (vit.getValue() == CellLabels::INTERIOR_CELL ||
				    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    hasSolvableCell = true;
			    return;
			}
		    }
		}
	    });

	    return hasSolvableCell;
	};	    

	assert(checkSolvableCell(myDomainCellLabels[0]));
	assert(unitTestBoundaryCells(myDomainCellLabels[0]));
	assert(unitTestExteriorCells(myDomainCellLabels[0]));

	// Precompute the coarsening strategy. Cap level if there are no longer interior cells
	for (int level = 1; level < myMGLevels; ++level)
	{
	    myDomainCellLabels[level] = buildCoarseCellLabels(myDomainCellLabels[level - 1]);

	    if (!checkSolvableCell(myDomainCellLabels[level]))
	    {
		myMGLevels = level - 1;
		myDomainCellLabels.setSize(myMGLevels);
		break;
	    }

	    assert(unitTestCoarsening(myDomainCellLabels[level], myDomainCellLabels[level - 1]));
	    assert(unitTestBoundaryCells(myDomainCellLabels[level]));
	    assert(unitTestExteriorCells(myDomainCellLabels[0]));
	}

	time = timer.stop();
	std::cout << "      Build coarse cell time: " << time << std::endl;
	timer.clear();
	timer.start();

	myDx.setSize(myMGLevels);
	myDx[0] = dx;
	    
	for (int level = 1; level < myMGLevels; ++level)
	    myDx[level] = 2. * myDx[level - 1];

	// Initialize solution vectors
	mySolutionGrids.setSize(myMGLevels);
	myRHSGrids.setSize(myMGLevels);
	myResidualGrids.setSize(myMGLevels);

	for (int level = 0; level < myMGLevels; ++level)
	{    
	    UT_Vector3I localRes = myDomainCellLabels[level].getVoxelRes();

	    mySolutionGrids[level].size(localRes[0], localRes[1], localRes[2]);
	    mySolutionGrids[level].constant(0);

	    myRHSGrids[level].size(localRes[0], localRes[1], localRes[2]);
	    myRHSGrids[level].constant(0);

	    myResidualGrids[level].size(localRes[0], localRes[1], localRes[2]);
	    myResidualGrids[level].constant(0);
	}

	myBoundaryCells.setSize(myMGLevels);
	for (int level = 0; level < myMGLevels; ++level)
	    myBoundaryCells[level] = buildBoundaryCells(myDomainCellLabels[level], myBoundarySmootherWidth);

	time = timer.stop();
	std::cout << "      Build boundary cells time: " << time << std::endl;
	timer.clear();
	timer.start();

	// Pre-build matrix at the coarsest level
	{
	    exint interiorCellCount = 0;
	    UT_Vector3I coarsestSize = myDomainCellLabels[myMGLevels - 1].getVoxelRes();

	    myDirectSolverIndices.size(coarsestSize[0], coarsestSize[1], coarsestSize[2]);
	    myDirectSolverIndices.constant(UNLABELLED_CELL);

	    {
		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&myDomainCellLabels[myMGLevels - 1]);
		UT_VoxelTileIterator<int> vitt;

		UT_Interrupt *boss = UTgetInterrupt();

		for (vit.rewind(); !vit.atEnd(); vit.advanceTile())
		{
		    if (boss->opInterrupt())
			break;

		    if (!vit.isTileConstant() ||
			vit.getValue() == CellLabels::INTERIOR_CELL ||
			vit.getValue() == CellLabels::BOUNDARY_CELL)
		    {
			vitt.setTile(vit);
			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL ||
				vitt.getValue() == CellLabels::BOUNDARY_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
				myDirectSolverIndices.setValue(cell, interiorCellCount++);
			    }
			}
		    }
		}
	    }

	    // Build rows for direct solver at coarsest level
	    const int threadCount = UT_Thread::getNumProcessors();

	    std::vector<std::vector<Eigen::Triplet<SolveReal>>> parallelSparseElements(threadCount);

	    SolveReal coarseGridScale = 1. / (myDx[myMGLevels - 1] * myDx[myMGLevels - 1]);

	    UT_ThreadedAlgorithm buildPoissonSystemAlgorithm;
	    buildPoissonSystemAlgorithm.run([&](const UT_JobInfo &info)
	    {
		UT_Interrupt *boss = UTgetInterrupt();

		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&myDomainCellLabels[myMGLevels - 1]);
		UT_VoxelTileIterator<int> vitt;

		std::vector<Eigen::Triplet<SolveReal>> &localSparseElements = parallelSparseElements[info.job()];

		for (vit.rewind(); !vit.atEnd(); vit.advanceTile())
		{
		    if (boss->opInterrupt())
			break;

		    if (!vit.isTileConstant() ||
			vit.getValue() == INTERIOR_CELL ||
			vit.getValue() == BOUNDARY_CELL)
		    {
			vitt.setTile(vit);
			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == INTERIOR_CELL ||
				vitt.getValue() == BOUNDARY_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				SolveReal diagonal = 0;
				exint index = myDirectSolverIndices(cell);
				assert(index >= 0);

				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

					auto adjacentCellLabel = myDomainCellLabels[myMGLevels - 1](adjacentCell);
					if (adjacentCellLabel == CellLabels::INTERIOR_CELL ||
					    adjacentCellLabel == CellLabels::BOUNDARY_CELL)
					{
					    exint adjacentIndex = myDirectSolverIndices(adjacentCell);
					    assert(adjacentIndex >= 0);

					    localSparseElements.push_back(Eigen::Triplet<SolveReal>(index, adjacentIndex, -coarseGridScale));
					    diagonal += coarseGridScale;
					}
					else if (adjacentCellLabel == CellLabels::DIRICHLET_CELL)
					    diagonal += coarseGridScale;
				    }

				localSparseElements.push_back(Eigen::Triplet<SolveReal>(index, index, diagonal));
			    }
			}
		    }
		}

		return 0;
	    });

	    exint listSize = 0;
	    for (int thread = 0; thread < threadCount; ++thread)
		listSize += parallelSparseElements[thread].size();

	    std::vector<Eigen::Triplet<SolveReal>> sparseElements;
	    sparseElements.reserve(listSize);

	    for (int thread = 0; thread < threadCount; ++thread)
	    {
		sparseElements.insert(sparseElements.end(), parallelSparseElements[thread].begin(), parallelSparseElements[thread].end());
		parallelSparseElements[thread].clear();
	    }

	    // Solve system
	    sparseMatrix = Eigen::SparseMatrix<SolveReal>(interiorCellCount, interiorCellCount);
	    sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
	    sparseMatrix.makeCompressed();

	    myCoarseSolver.compute(sparseMatrix);

	    assert(myCoarseSolver.info() == Eigen::Success);

	    myCoarseRHSVector = Vector::Zero(interiorCellCount);
	}

	time = timer.stop();
	std::cout << "      Build coarse direct solver time: " << time << std::endl;
	timer.clear();
	timer.start();
    }

    void
    GeometricMultigridPoissonSolver::applyVCycle(UT_VoxelArray<StoreReal> &fineSolutionGrid,
						    const UT_VoxelArray<StoreReal> &fineRHSGrid,
						    const bool useInitialGuess)
    {
	using namespace HDK::GeometricMultigridOperators;

	assert(fineSolutionGrid.getVoxelRes() == fineRHSGrid.getVoxelRes() ||
		fineRHSGrid.getVoxelRes() == myDomainCellLabels[0].getVoxelRes());

	{
	    UT_StopWatch precookTimer;
	    precookTimer.start();

	    if (!useInitialGuess)
		fineSolutionGrid.constant(0);

	    uncompressBoundaryTiles(fineSolutionGrid, myBoundaryCells[0]);

	    auto time = precookTimer.stop();
	    std::cout << "      V-cycle pre-cook time: " << time << std::endl;
	}
	
	// Apply fine-level smoothing pass
	{
	    std::cout << "    Fine Downstroke Smoother" << std::endl;

	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    boundaryJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
								fineRHSGrid,
								myDomainCellLabels[0],
								myBoundaryCells[0],
								myDx[0],
								&myFineBoundaryWeights);
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch smoothTimer;
		smoothTimer.start();

		// Apply smoother
		if (myUseGaussSeidel)
		{
		    interiorTiledGaussSeidelPoissonSmoother<SolveReal>(fineSolutionGrid,
									fineRHSGrid,
									myDomainCellLabels[0],
									myDx[0],
									true /*smooth odd tiles*/, true /*smooth forwards*/);

		    interiorTiledGaussSeidelPoissonSmoother<SolveReal>(fineSolutionGrid,
									fineRHSGrid,
									myDomainCellLabels[0],
									myDx[0],
									false /*smooth even tiles*/, true /*smooth forwards*/);
		}
		else
		{
		    interiorJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
								fineRHSGrid,
								myDomainCellLabels[0],
								myDx[0]);
		}

		auto time = smoothTimer.stop();
		std::cout << "      Smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    boundaryJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
								fineRHSGrid,
								myDomainCellLabels[0],
								myBoundaryCells[0],
								myDx[0],
								&myFineBoundaryWeights);
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch computeResidualTimer;
		computeResidualTimer.start();

		// Compute residual to restrict to the next level
		computePoissonResidual<SolveReal>(myResidualGrids[0],
						    fineSolutionGrid,
						    fineRHSGrid,
						    myDomainCellLabels[0],
						    myDx[0],
						    &myFineBoundaryWeights);

		auto time = computeResidualTimer.stop();
		std::cout << "      Compute residual time: " << time << std::endl;
	    }

	    {
		UT_StopWatch restrictionTimer;
		restrictionTimer.start();

		downsample<SolveReal>(myRHSGrids[1],
					myResidualGrids[0],
					myDomainCellLabels[1],
					myDomainCellLabels[0]);

		auto time = restrictionTimer.stop();
		std::cout << "      Restriction time: " << time << std::endl;
	    }

	    {
		UT_StopWatch cleanUpTimer;
		cleanUpTimer.start();

		mySolutionGrids[1].constant(0);

		// Expand tiles at boundaries
		uncompressBoundaryTiles(mySolutionGrids[1], myBoundaryCells[1]);

		auto time = cleanUpTimer.stop();
		std::cout << "      Clean up time: " << time << std::endl;
	    }
	}

	// Apply course-level smoothing pass
	for (int level = 1; level < myMGLevels - 1; ++level)
	{
	    std::cout << "    Downstroke Smoother level: " << level << std::endl;

	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								myRHSGrids[level],
								myDomainCellLabels[level],
								myBoundaryCells[level],
								myDx[level]);
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch smoothTimer;
		smoothTimer.start();

		// Apply smoother
		if (myUseGaussSeidel)
		{
		    interiorTiledGaussSeidelPoissonSmoother<SolveReal>(mySolutionGrids[level],
									myRHSGrids[level],
									myDomainCellLabels[level],
									myDx[level],
									true /*smooth odd tiles*/, true /*smooth forwards*/);

		    interiorTiledGaussSeidelPoissonSmoother<SolveReal>(mySolutionGrids[level],
									myRHSGrids[level],
									myDomainCellLabels[level],
									myDx[level],
									false /*smooth even tiles*/, true /*smooth forwards*/);
		}
		else
		{
		    interiorJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								myRHSGrids[level],
								myDomainCellLabels[level],
								myDx[level]);
		}

		auto time = smoothTimer.stop();
		std::cout << "      Smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								myRHSGrids[level],
								myDomainCellLabels[level],
								myBoundaryCells[level],
								myDx[level]);
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch computeResidualTimer;
		computeResidualTimer.start();

		// Compute residual to restrict to the next level
		computePoissonResidual<SolveReal>(myResidualGrids[level],
						    mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myDx[level]);

		auto time = computeResidualTimer.stop();
		std::cout << "      Compute residual time: " << time << std::endl;
	    }

	    {
		UT_StopWatch restrictionTimer;
		restrictionTimer.start();

		downsample<SolveReal>(myRHSGrids[level + 1],
					myResidualGrids[level],
					myDomainCellLabels[level + 1],
					myDomainCellLabels[level]);

		auto time = restrictionTimer.stop();
		std::cout << "      Restriction time: " << time << std::endl;
	    }

	    {
		UT_StopWatch cleanUpTimer;
		cleanUpTimer.start();

		mySolutionGrids[level + 1].constant(0);

		// Expand tiles at boundaries
		if (level < myMGLevels - 1)
		    uncompressBoundaryTiles(mySolutionGrids[level + 1], myBoundaryCells[level + 1]);


		auto time = cleanUpTimer.stop();
		std::cout << "      Clean up time: " << time << std::endl;
	    }
	}

	{
	    UT_StopWatch directSolveTimer;
	    directSolveTimer.start();

	    copyGridToVector(myCoarseRHSVector,
				myRHSGrids[myMGLevels - 1],
				myDirectSolverIndices,
				myDomainCellLabels[myMGLevels - 1]);

	    Vector directSolution = myCoarseSolver.solve(myCoarseRHSVector);

	    copyVectorToGrid(mySolutionGrids[myMGLevels - 1],
				directSolution,
				myDirectSolverIndices,
				myDomainCellLabels[myMGLevels - 1]);

	    auto time = directSolveTimer.stop();
	    std::cout << "      Direct solve time: " << time << std::endl;
	}

	// Up-stroke of the v-cycle
	for (int level = myMGLevels - 2; level >= 1; --level)
	{
	    std::cout << "    Upstroke Smoother level: " << level << std::endl;
	    {
		UT_StopWatch prolongationTimer;
		prolongationTimer.start();
		
		upsampleAndAdd<SolveReal>(mySolutionGrids[level],
					    mySolutionGrids[level + 1],
					    myDomainCellLabels[level],
					    myDomainCellLabels[level + 1]);

		auto time = prolongationTimer.stop();
		std::cout << "      Prolongation time: " << time << std::endl;
	    }

	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								myRHSGrids[level],
								myDomainCellLabels[level],
								myBoundaryCells[level],
								myDx[level]);
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch smoothTimer;
		smoothTimer.start();

		// Smooth interior
		if (myUseGaussSeidel)
		{
		    interiorTiledGaussSeidelPoissonSmoother<SolveReal>(mySolutionGrids[level],
									myRHSGrids[level],
									myDomainCellLabels[level],
									myDx[level],
									false /*smooth even tiles*/, false /*smooth backwards*/);

		    interiorTiledGaussSeidelPoissonSmoother<SolveReal>(mySolutionGrids[level],
									myRHSGrids[level],
									myDomainCellLabels[level],
									myDx[level],
									true /*smooth odd tiles*/, false /*smooth backwards*/);
		}
		else
		{
		    interiorJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								myRHSGrids[level],
								myDomainCellLabels[level],
								myDx[level]);
		}

		auto time = smoothTimer.stop();
		std::cout << "      Smoother time: " << time << std::endl;
	    }
	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								myRHSGrids[level],
								myDomainCellLabels[level],
								myBoundaryCells[level],
								myDx[level]);
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }
	}

	// Apply fine-level upstroke
    	{
	    std::cout << "    Fine Upstroke Smoother" << std::endl;
	    {
		UT_StopWatch prolongationTimer;
		prolongationTimer.start();
		
		upsampleAndAdd<SolveReal>(fineSolutionGrid,
					    mySolutionGrids[1],
					    myDomainCellLabels[0],
					    myDomainCellLabels[1]);

		auto time = prolongationTimer.stop();
		std::cout << "      Prolongation time: " << time << std::endl;
	    }

	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    boundaryJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
								fineRHSGrid,
								myDomainCellLabels[0],
								myBoundaryCells[0],
								myDx[0],
								&myFineBoundaryWeights);
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch smoothTimer;
		smoothTimer.start();

		// Smooth interior
		if (myUseGaussSeidel)
		{
		    interiorTiledGaussSeidelPoissonSmoother<SolveReal>(fineSolutionGrid,
									fineRHSGrid,
									myDomainCellLabels[0],
									myDx[0],
									false /*smooth even tiles*/, false /*smooth backwards*/);

		    interiorTiledGaussSeidelPoissonSmoother<SolveReal>(fineSolutionGrid,
									fineRHSGrid,
									myDomainCellLabels[0],
									myDx[0],
									true /*smooth odd tiles*/, false /*smooth backwards*/);
		}
		else
		{
		    interiorJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
								fineRHSGrid,
								myDomainCellLabels[0],
								myDx[0]);
		}

		auto time = smoothTimer.stop();
		std::cout << "      Smoother time: " << time << std::endl;
	    }
	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    boundaryJacobiPoissonSmoother<SolveReal>(fineSolutionGrid,
								fineRHSGrid,
								myDomainCellLabels[0],
								myBoundaryCells[0],
								myDx[0],
								&myFineBoundaryWeights);
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }
	}
    }
}