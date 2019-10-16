#include "HDK_GeometricMultigridPoissonSolver.h"

#include <SIM/SIM_FieldUtils.h>
#include <UT/UT_ParallelUtil.h>

#include <UT/UT_StopWatch.h>

#include "HDK_GeometricMultigridOperators.h"

namespace HDK
{
    //
    // Helper functions for multigrid
    //

    template<typename StoreReal>
    void
    copyToExpandedGrid(UT_VoxelArray<StoreReal> &expandedDestination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &expandedCellLabels,
			const UT_Vector3I &expandedOffset)
    {
	using namespace HDK::GeometricMultigridOperators;

	UT_Interrupt *boss = UTgetInterrupt();

#if !defined(NDEBUG)
	for (int axis : {0,1,2})
	    assert(source.getVoxelRes()[axis] + 2 * expandedOffset[axis] <= expandedDestination.getVoxelRes()[axis]);
#endif
	assert(expandedDestination.getVoxelRes() == expandedCellLabels.getVoxelRes());

	// Uncompress tiles that will be written to
	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(expandedCellLabels.numTiles());
	isTileOccupiedList.constant(false);

	UTparallelForEachNumber(expandedCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&expandedCellLabels);
	    UT_VoxelTileIterator<int> vitt;

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.atEnd())
		{
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
				UT_Vector3I cell = UT_Vector3I(vitt.x(), vitt.y(), vitt.z());
				int tileNumber = expandedDestination.indexToLinearTile(cell[0], cell[1], cell[2]);

				if (!isTileOccupiedList[tileNumber])
				    isTileOccupiedList[tileNumber] = true;
			    }
			}
		    }
		}
	    }
	});

	uncompressTiles(expandedDestination, isTileOccupiedList);

	UTparallelForEachNumber(expandedCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&expandedCellLabels);

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> destinationProbe;
	    destinationProbe.setArray(&expandedDestination);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> sourceProbe;
	    sourceProbe.setConstArray(&source);

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.atEnd())
		{
		    if (!vit.isTileConstant() ||
			vit.getValue() == CellLabels::INTERIOR_CELL ||
			vit.getValue() == CellLabels::BOUNDARY_CELL)
		    {
			for (; !vit.atEnd(); vit.advance())
			{
			    if (vit.getValue() == CellLabels::INTERIOR_CELL ||
				vit.getValue() == CellLabels::BOUNDARY_CELL)
			    {
				UT_Vector3I expandedCell(vit.x(), vit.y(), vit.z());
				assert(!expandedDestination.getLinearTile(expandedDestination.indexToLinearTile(expandedCell[0],
														expandedCell[1],
														expandedCell[2]))->isConstant());

				UT_Vector3I cell = expandedCell - expandedOffset;

				assert(cell[0] >= 0 && cell[1] >= 0 && cell[2] >= 0 &&
					cell[0] < source.getVoxelRes()[0] &&
					cell[1] < source.getVoxelRes()[1] &&
					cell[2] < source.getVoxelRes()[2]);

				destinationProbe.setIndex(vit);
				sourceProbe.setIndex(cell[0], cell[1], cell[2]);
				destinationProbe.setValue(sourceProbe.getValue());
			    }
			}
		    }
		}
	    }
	});
    }

    template<typename StoreReal>
    void
    copyFromExpandedGrid(UT_VoxelArray<StoreReal> &destination,
			    const UT_VoxelArray<StoreReal> &expandedSource,
			    const UT_VoxelArray<int> &expandedCellLabels,
			    const UT_Vector3I &expandedOffset)
    {
	using namespace HDK::GeometricMultigridOperators;

	UT_Interrupt *boss = UTgetInterrupt();

#if !defined(NDEBUG)
	for (int axis : {0,1,2})
	    assert(destination.getVoxelRes()[axis] + 2 * expandedOffset[axis] <= expandedSource.getVoxelRes()[axis]);
#endif
	assert(expandedSource.getVoxelRes() == expandedCellLabels.getVoxelRes());

	// Uncompress tiles that will be written to
	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(destination.numTiles());
	isTileOccupiedList.constant(false);

	UTparallelForEachNumber(expandedCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&expandedCellLabels);
	    UT_VoxelTileIterator<int> vitt;

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.atEnd())
		{
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
				UT_Vector3I expandedCell(vitt.x(), vitt.y(), vitt.z());
				UT_Vector3I cell = expandedCell - expandedOffset;

				assert(cell[0] >= 0 && cell[1] >= 0 && cell[2] >= 0 &&
					cell[0] < destination.getVoxelRes()[0] &&
					cell[1] < destination.getVoxelRes()[1] &&
					cell[2] < destination.getVoxelRes()[2]);

				int tileNumber = destination.indexToLinearTile(cell[0], cell[1], cell[2]);

				if (!isTileOccupiedList[tileNumber])
				    isTileOccupiedList[tileNumber] = true;
			    }
			}
		    }
		}
	    }
	});

	uncompressTiles(destination, isTileOccupiedList);

	UTparallelForEachNumber(expandedCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&expandedCellLabels);

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> destinationProbe;
	    destinationProbe.setArray(&destination);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> sourceProbe;
	    sourceProbe.setConstArray(&expandedSource);

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.atEnd())
		{
		    if (!vit.isTileConstant() ||
			vit.getValue() == CellLabels::INTERIOR_CELL ||
			vit.getValue() == CellLabels::BOUNDARY_CELL)
		    {
			for (; !vit.atEnd(); vit.advance())
			{
			    if (vit.getValue() == CellLabels::INTERIOR_CELL ||
				vit.getValue() == CellLabels::BOUNDARY_CELL)
			    {
				UT_Vector3I expandedCell(vit.x(), vit.y(), vit.z());
				UT_Vector3I cell = expandedCell - expandedOffset;

				assert(cell[0] >= 0 && cell[1] >= 0 && cell[2] >= 0 &&
					cell[0] < destination.getVoxelRes()[0] &&
					cell[1] < destination.getVoxelRes()[1] &&
					cell[2] < destination.getVoxelRes()[2]);

				assert(!destination.getLinearTile(destination.indexToLinearTile(cell[0],
												cell[1],
												cell[2]))->isConstant());

				destinationProbe.setIndex(cell[0], cell[1], cell[2]);
				sourceProbe.setIndex(vit);
				destinationProbe.setValue(sourceProbe.getValue());
			    }
			}
		    }
		}
	    }
	});
    }

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
									const int mgLevels,
									const SolveReal dx,
									const int boundarySmootherWidth,
									const int boundarySmootherIterations,
									const bool useGaussSeidel)
    : myDoApplyBoundaryWeights(false)
    , myMGLevels(mgLevels)
    , myBoundarySmootherWidth(boundarySmootherWidth)
    , myBoundarySmootherIterations(boundarySmootherIterations)
    , myUseGaussSeidel(useGaussSeidel)
    {
	using namespace HDK::GeometricMultigridOperators;
	using namespace SIM::FieldUtils;

	assert(myMGLevels > 0);
	assert(dx > 0);

	UT_StopWatch timer;
	timer.start();

	// Add the necessary exterior cells so that after coarsening to the top level
	// there is still a single layer of exterior cells
	int exteriorPadding = std::pow(2, myMGLevels - 1);

	UT_Vector3I expandedResolution = initialDomainCellLabels.getVoxelRes() + 2 * UT_Vector3I(exteriorPadding);

	// Expand the domain to be a power of 2.
	for (int axis : {0,1,2})
	{
	    fpreal logSize = std::log2(fpreal(expandedResolution[axis]));
	    logSize = std::ceil(logSize);

	    expandedResolution[axis] = exint(std::exp2(logSize));
	}
	
	myExteriorOffset = UT_Vector3I(exteriorPadding);

	// Clamp top level to the highest possible coarsening given the resolution.
	for (int axis : {0,1,2})
	{
	    int topLevel = int(std::log2(expandedResolution[axis]));
	    
	    if (topLevel < myMGLevels)
		myMGLevels = topLevel;
	}

	// Build finest level domain labels with the necessary padding to maintain a 
	// 1-band ring of exterior cells at the coarsest level.
	myDomainCellLabels.setSize(myMGLevels);
	myDomainCellLabels[0].size(expandedResolution[0], expandedResolution[1], expandedResolution[2]);
	myDomainCellLabels[0].constant(CellLabels::EXTERIOR_CELL);

	UT_Interrupt *boss = UTgetInterrupt();

	{
	    // Uncompress internal domain label tiles
	    UT_Array<bool> isTileOccupiedList;
	    isTileOccupiedList.setSize(myDomainCellLabels[0].numTiles());
	    isTileOccupiedList.constant(false);

	    UTparallelForEachNumber(initialDomainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&initialDomainCellLabels);
		UT_VoxelTileIterator<int> vitt;

		for (int i = range.begin(); i != range.end(); ++i)
		{
		    vit.myTileStart = i;
		    vit.myTileEnd = i + 1;
		    vit.rewind();

		    if (boss->opInterrupt())
			break;

		    if (!vit.atEnd())
		    {
			if (!vit.isTileConstant() || vit.getValue() != CellLabels::EXTERIOR_CELL)
			{
			    vitt.setTile(vit);

			    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			    {
				if (vitt.getValue() != CellLabels::EXTERIOR_CELL)
				{
				    UT_Vector3I cell = UT_Vector3I(vitt.x(), vitt.y(), vitt.z()) + myExteriorOffset;
				    int tileNumber = myDomainCellLabels[0].indexToLinearTile(cell[0], cell[1], cell[2]);
				    if (!isTileOccupiedList[tileNumber])
					isTileOccupiedList[tileNumber] = true;
				}
			    }
			}
		    }
		}
	    });

	    uncompressTiles(myDomainCellLabels[0], isTileOccupiedList);

	    // Copy initial domain labels to interior domain labels with padding
	    UTparallelForEachNumber(initialDomainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&initialDomainCellLabels);

		UT_VoxelProbe<int, true /* read */, false /* no write */, false> initialDomainProbe;
		initialDomainProbe.setConstArray(&initialDomainCellLabels);

		UT_VoxelProbe<int, false /* no read */, true /* write */, true /* test for write */> localDomainProbe;
		localDomainProbe.setArray(&myDomainCellLabels[0]);

		for (int i = range.begin(); i != range.end(); ++i)
		{
		    vit.myTileStart = i;
		    vit.myTileEnd = i + 1;
		    vit.rewind();

		    if (boss->opInterrupt())
			break;

		    if (!vit.atEnd())
		    {
			if (!vit.isTileConstant() ||
			    vit.getValue() != CellLabels::EXTERIOR_CELL)
			{
			    for (; !vit.atEnd(); vit.advance())
			    {
				if (vit.getValue() != CellLabels::EXTERIOR_CELL)
				{
				    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
				    UT_Vector3I expandedCell = cell + myExteriorOffset;

				    assert(!myDomainCellLabels[0].getLinearTile(myDomainCellLabels[0].indexToLinearTile(expandedCell[0], expandedCell[1], expandedCell[2]))->isConstant());

				    initialDomainProbe.setIndex(vit);
				    localDomainProbe.setIndex(expandedCell[0], expandedCell[1], expandedCell[2]);
				    localDomainProbe.setValue(initialDomainProbe.getValue());

				}
			    }
			}
		    }
		}
	    });

	    myDomainCellLabels[0].collapseAllTiles();
	}

	auto time = timer.stop();
	std::cout << "      Copy initial domain time: " << time << std::endl;
	timer.clear();
	timer.start();

	auto checkSolvableCell = [&](const UT_VoxelArray<int> &testGrid) -> bool
	{
	    bool hasSolvableCell = false;

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

    // TODO: Optimize this to only copy at faces between boundary and exterior or dirichlet cells
    void
    GeometricMultigridPoissonSolver::setBoundaryWeights(const std::array<UT_VoxelArray<StoreReal>, 3> &boundaryWeights)
    {
	myDoApplyBoundaryWeights = true;

	UT_Vector3I baseVoxelRes = myDomainCellLabels[0].getVoxelRes();
	for (int axis : {0,1,2})
	{
	    UT_Vector3I localVoxelRes = baseVoxelRes;
	    ++localVoxelRes[axis];

	    myFineBoundaryWeights[axis].size(localVoxelRes[0], localVoxelRes[1], localVoxelRes[2]);

	    UT_Interrupt *boss = UTgetInterrupt();

	    UTparallelForEachNumber(myFineBoundaryWeights[axis].numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<StoreReal> vit;
		vit.setConstArray(&myFineBoundaryWeights[axis]);
		UT_VoxelTileIterator<StoreReal> vitt;

		for (int i = range.begin(); i != range.end(); ++i)
		{
		    vit.myTileStart = i;
		    vit.myTileEnd = i + 1;
		    vit.rewind();

		    if (boss->opInterrupt())
			break;

		    if (!vit.atEnd())
		    {
			// TODO: handle constant tiles
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());
			    face -= myExteriorOffset;

			    // It's possible to step outside of the bounds of the provided gradient weights
			    if (face[0] < 0 || face[1] < 0 || face[2] < 0 ||
				face[0] >= boundaryWeights[axis].getVoxelRes()[0] ||
				face[1] >= boundaryWeights[axis].getVoxelRes()[1] ||
				face[2] >= boundaryWeights[axis].getVoxelRes()[2])
				continue;

			    vitt.setValue(boundaryWeights[axis](face));
			}
		    }
		}
	    });
	}
    }

    void
    GeometricMultigridPoissonSolver::applyVCycle(UT_VoxelArray<StoreReal> &solution,
						    const UT_VoxelArray<StoreReal> &rhs,
						    const bool useInitialGuess)
    {
	using namespace HDK::GeometricMultigridOperators;

#if !defined(NDEBUG)
	for (int axis : {0,1,2})
	{
	    assert(solution.getVoxelRes()[axis] < mySolutionGrids[0].getVoxelRes()[axis] &&
		    rhs.getVoxelRes()[axis] < mySolutionGrids[0].getVoxelRes()[axis]);
	}
#endif

	assert(solution.getVoxelRes() == rhs.getVoxelRes());

	{
	    UT_StopWatch precookTimer;
	    precookTimer.start();

	    mySolutionGrids[0].constant(0);
	    uncompressBoundaryTiles(mySolutionGrids[0], myBoundaryCells[0]);

	    myRHSGrids[0].constant(0);

	    // If there is an initial guess in the solution vector, copy it locally
	    if (useInitialGuess)
	    {
		copyToExpandedGrid(mySolutionGrids[0],
				    solution,
				    myDomainCellLabels[0],
				    myExteriorOffset);
	    }

	    // Copy RHS to internal expanded grid
	    copyToExpandedGrid(myRHSGrids[0],
				rhs,
				myDomainCellLabels[0],
				myExteriorOffset);

	    auto time = precookTimer.stop();
	    std::cout << "      V-cycle pre-cook time: " << time << std::endl;
	}

	// Down-stroke of the v-cycle
	for (int level = 0; level < myMGLevels - 1; ++level)
	{

	    std::cout << "    Downstroke Smoother level: " << level << std::endl;
	    
	    {
		UT_StopWatch boundarySmoothTimer;
		boundarySmoothTimer.start();

		// Smooth along boundaries
		for (int boundaryIteration = 0; boundaryIteration < myBoundarySmootherIterations; ++boundaryIteration)
		{
		    if (level == 0 && myDoApplyBoundaryWeights)
		    {
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								    myRHSGrids[level],
								    myDomainCellLabels[level],
								    myBoundaryCells[level],
								    myDx[level],
								    &myFineBoundaryWeights);
		    }
		    else
		    {
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								    myRHSGrids[level],
								    myDomainCellLabels[level],
								    myBoundaryCells[level],
								    myDx[level]);
		    }
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
		    if (level == 0 && myDoApplyBoundaryWeights)
		    {
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								    myRHSGrids[level],
								    myDomainCellLabels[level],
								    myBoundaryCells[level],
								    myDx[level],
								    &myFineBoundaryWeights);
		    }
		    else
		    {
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								    myRHSGrids[level],
								    myDomainCellLabels[level],
								    myBoundaryCells[level],
								    myDx[level]);
		    }
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }

	    {
		UT_StopWatch computeResidualTimer;
		computeResidualTimer.start();

		// Compute residual to restrict to the next level
		if (level == 0 && myDoApplyBoundaryWeights)
		{
		    computePoissonResidual<SolveReal>(myResidualGrids[level],
							mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							&myFineBoundaryWeights);
		}
		else
		{
		    computePoissonResidual<SolveReal>(myResidualGrids[level],
							mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level]);
		}

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
	for (int level = myMGLevels - 2; level >= 0; --level)
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
		    if (level == 0 && myDoApplyBoundaryWeights)
		    {
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								    myRHSGrids[level],
								    myDomainCellLabels[level],
								    myBoundaryCells[level],
								    myDx[level],
								    &myFineBoundaryWeights);
		    }
		    else
		    {
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								    myRHSGrids[level],
								    myDomainCellLabels[level],
								    myBoundaryCells[level],
								    myDx[level]);
		    }
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
		    if (level == 0 && myDoApplyBoundaryWeights)
		    {
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								    myRHSGrids[level],
								    myDomainCellLabels[level],
								    myBoundaryCells[level],
								    myDx[level],
								    &myFineBoundaryWeights);
		    }
		    else
		    {
			boundaryJacobiPoissonSmoother<SolveReal>(mySolutionGrids[level],
								    myRHSGrids[level],
								    myDomainCellLabels[level],
								    myBoundaryCells[level],
								    myDx[level]);
		    }
		}

		auto time = boundarySmoothTimer.stop();
		std::cout << "      Boundary smoother time: " << time << std::endl;
	    }
	    
	}

	{
	    UT_StopWatch copySmootherTimer;
	    copySmootherTimer.start();

	    // Copy local solution vector with expanded exterior band to 
	    // an interior solution that matches the supplied RHS vector grid.
	    copyFromExpandedGrid(solution,
				    mySolutionGrids[0],
				    myDomainCellLabels[0],
				    myExteriorOffset);

	    auto time = copySmootherTimer.stop();
	    std::cout << "      Copy solution time: " << time << std::endl;
	}
    }
}