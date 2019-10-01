#include "HDK_GeometricMultiGridPoissonSolver.h"

#include <SIM/SIM_FieldUtils.h>
#include <UT/UT_ParallelUtil.h>

#include "HDK_GeometricMultiGridOperations.h"

namespace HDK
{
    template<typename GridType>
    void uncompressTile(UT_VoxelArray<GridType> &grid,
			const UT_Array<bool> &isTileOccupiedList)
    {
	UT_Interrupt *boss = UTgetInterrupt();
	UTparallelFor(UT_BlockedRange<exint>(0, isTileOccupiedList.size()), [&](const UT_BlockedRange<exint> &range)
	{
	    for (exint i = range.begin(); i != range.end(); ++i)
	    {
		if (!(i & 127))
		{
		    if (boss->opInterrupt())
			return;
		}

		if (isTileOccupiedList[i])
		    grid.getLinearTile(i)->uncompress();
	    }
	});
    }

    GeometricMultiGridPoissonSolver::GeometricMultiGridPoissonSolver(const UT_VoxelArray<int> &initialDomainCellLabels,
									const int mgLevels,
									const SolveReal dx,
									const int boundarySmootherWidth,
									const int boundarySmootherIterations,
									const int smootherIterations,
									const bool useGaussSeidel)
    : myDoApplyGradientWeights(false)
    , myMGLevels(mgLevels)
    , myBoundarySmootherWidth(boundarySmootherWidth)
    , myBoundarySmootherIterations(boundarySmootherIterations)
    , myTotalSmootherIterations(smootherIterations)
    , myUseGaussSeidel(useGaussSeidel)
    {
	using namespace HDK::GeometricMultiGridOperations;
	using namespace SIM::FieldUtils;

	assert(myMGLevels > 0);
	assert(dx > 0);

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

	    uncompressTile(myDomainCellLabels[0], isTileOccupiedList);

	    // Copy initial domain labels to interior domain labels with padding
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
				    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
				    UT_Vector3I expandedCell = cell + myExteriorOffset;

				    assert(!myDomainCellLabels[0].getLinearTile(myDomainCellLabels[0].indexToLinearTile(expandedCell[0], expandedCell[1], expandedCell[2]))->isConstant());

				    myDomainCellLabels[0].setValue(expandedCell, initialDomainCellLabels(cell));
				}
			    }
			}
		    }
		}
	    });

	    myDomainCellLabels[0].collapseAllTiles();
	}

	auto checkInteriorCell = [&](const UT_VoxelArray<int> &testGrid) -> bool
	{
	    bool hasInteriorCell = false;

	    UTparallelForEachNumber(testGrid.numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		if (hasInteriorCell) return;

		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&testGrid);
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
			if (!vit.isTileConstant())
			{
			    vitt.setTile(vit);

			    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			    {
				if (vitt.getValue() == CellLabels::INTERIOR_CELL)
				{
				    hasInteriorCell = true;
				    return;
				}
			    }
			}
			else if (vit.getValue() == CellLabels::INTERIOR_CELL)
			{
			    hasInteriorCell = true;
			    return;
			}
		    }
		}
	    });

	    return hasInteriorCell;
	};	    

	assert(checkInteriorCell(myDomainCellLabels[0]));

	// Precompute the coarsening strategy. Cap level if there are no longer interior cells
	for (int level = 1; level < myMGLevels; ++level)
	{
	    myDomainCellLabels[level] = buildCoarseCellLabels(myDomainCellLabels[level - 1]);

	    if (!checkInteriorCell(myDomainCellLabels[level]))
	    {
		myMGLevels = level - 1;
		myDomainCellLabels.setSize(myMGLevels);
		break;
	    }
	}

	myDx.setSize(myMGLevels);
	myDx[0] = dx;
	    
	for (int level = 1; level < myMGLevels; ++level)
	    myDx[level] = 2. * myDx[level - 1];

	myBoundaryCells.setSize(myMGLevels);
	for (int level = 0; level < myMGLevels; ++level)
	    myBoundaryCells[level] = buildBoundaryCells(myDomainCellLabels[level], myBoundarySmootherWidth);

	// Initialize solution vectors
	mySolutionGrids.setSize(myMGLevels);
	myRHSGrids.setSize(myMGLevels);
	myResidualGrids.setSize(myMGLevels);

	for (int level = 0; level < myMGLevels; ++level)
	{    
	    UT_Vector3I localRes = myDomainCellLabels[level].getVoxelRes();

	    mySolutionGrids[level].size(localRes[0], localRes[1], localRes[2]);
	    myRHSGrids[level].size(localRes[0], localRes[1], localRes[2]);
	    myResidualGrids[level].size(localRes[0], localRes[1], localRes[2]);
	}

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

		    if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);
			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == INTERIOR_CELL)
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

		    if (!vit.isTileConstant() || vit.getValue() == INTERIOR_CELL)
		    {
			vitt.setTile(vit);
			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == INTERIOR_CELL)
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
					if (adjacentCellLabel == CellLabels::INTERIOR_CELL)
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
    }

    void
    GeometricMultiGridPoissonSolver::setGradientWeights(const UT_VoxelArray<StoreReal> (&gradientWeights)[3])
    {
	myDoApplyGradientWeights = true;

	UT_Vector3I baseVoxelRes = myDomainCellLabels[0].getVoxelRes();
	for (int axis : {0,1,2})
	{
	    UT_Vector3I localVoxelRes = baseVoxelRes;
	    ++localVoxelRes[axis];

	    myFineGradientWeights[axis].size(localVoxelRes[0], localVoxelRes[1], localVoxelRes[2]);

	    UT_Interrupt *boss = UTgetInterrupt();

	    UTparallelForEachNumber(myFineGradientWeights[axis].numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<StoreReal> vit;
		vit.setConstArray(&myFineGradientWeights[axis]);
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
				face[0] >= gradientWeights[axis].getVoxelRes()[0] ||
				face[1] >= gradientWeights[axis].getVoxelRes()[1] ||
				face[2] >= gradientWeights[axis].getVoxelRes()[2])
				continue;

			    vitt.setValue(gradientWeights[axis](face));
			}
		    }
		}
	    });
	}
    }

    template<typename StoreReal>
    void
    copyToExpandedGrid(UT_VoxelArray<StoreReal> &expandedDestination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &expandedCellLabels,
			const UT_Vector3I &expandedOffset)
    {
	using namespace HDK::GeometricMultiGridOperations;

	UT_Interrupt *boss = UTgetInterrupt();

	for (int axis : {0,1,2})
	    assert(source.getVoxelRes()[axis] + 2 * expandedOffset[axis] <= expandedDestination.getVoxelRes()[axis]);

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
		    if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
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

	uncompressTile(expandedDestination, isTileOccupiedList);

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
		    if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I expandedCell(vitt.x(), vitt.y(), vitt.z());
				assert(!expandedDestination.getLinearTile(expandedDestination.indexToLinearTile(expandedCell[0],
														expandedCell[1],
														expandedCell[2]))->isConstant());

				UT_Vector3I cell = expandedCell - expandedOffset;

				assert(cell[0] >= 0 && cell[1] >= 0 && cell[2] >= 0 &&
					cell[0] < source.getVoxelRes()[0] &&
					cell[1] < source.getVoxelRes()[1] &&
					cell[2] < source.getVoxelRes()[2]);

				expandedDestination.setValue(expandedCell, source(cell));
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
	using namespace HDK::GeometricMultiGridOperations;

	UT_Interrupt *boss = UTgetInterrupt();

	for (int axis : {0,1,2})
	    assert(destination.getVoxelRes()[axis] + 2 * expandedOffset[axis] <= expandedSource.getVoxelRes()[axis]);

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
		    if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
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

	uncompressTile(destination, isTileOccupiedList);

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
		    if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I expandedCell(vitt.x(), vitt.y(), vitt.z());
				UT_Vector3I cell = expandedCell - expandedOffset;

				assert(cell[0] >= 0 && cell[1] >= 0 && cell[2] >= 0 &&
					cell[0] < destination.getVoxelRes()[0] &&
					cell[1] < destination.getVoxelRes()[1] &&
					cell[2] < destination.getVoxelRes()[2]);

				assert(!destination.getLinearTile(destination.indexToLinearTile(cell[0],
												cell[1],
												cell[2]))->isConstant());

				destination.setValue(cell, expandedSource(expandedCell));
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
	using namespace HDK::GeometricMultiGridOperations;

	UT_Interrupt *boss = UTgetInterrupt();

	assert(gridVector.getVoxelRes() == gridIndices.getVoxelRes() &&
		gridIndices.getVoxelRes() == cellLabels.getVoxelRes());

	UTparallelForEachNumber(gridIndices.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<exint> vit;
	    vit.setConstArray(&gridIndices);
	    UT_VoxelTileIterator<exint> vitt;

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
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			    exint index = vitt.getValue();
			    if (index >= 0)
			    {
				assert(cellLabels(cell) == CellLabels::INTERIOR_CELL);
				vector(index) = gridVector(cell);
			    }
			    else assert(cellLabels(cell) != CellLabels::INTERIOR_CELL);
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
	using namespace HDK::GeometricMultiGridOperations;

	UT_Interrupt *boss = UTgetInterrupt();

	assert(gridVector.getVoxelRes() == gridIndices.getVoxelRes() &&
		gridIndices.getVoxelRes() == cellLabels.getVoxelRes());

	UTparallelForEachNumber(gridIndices.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<exint> vit;
	    vit.setConstArray(&gridIndices);
	    UT_VoxelTileIterator<exint> vitt;

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
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			    exint index = vitt.getValue();
			    if (index >= 0)
			    {
				assert(cellLabels(cell) == CellLabels::INTERIOR_CELL);
				gridVector.setValue(cell, vector(index));
			    }
			    else assert(cellLabels(cell) != CellLabels::INTERIOR_CELL);
			}
		    }
		}
	    }
	});
    }

    void
    GeometricMultiGridPoissonSolver::applyVCycle(UT_VoxelArray<StoreReal> &solution,
						    UT_VoxelArray<StoreReal> &rhs,
						    const bool useInitialGuess)
    {
	using namespace HDK::GeometricMultiGridOperations;

	for (int axis : {0,1,2})
	{
	    assert(solution.getVoxelRes()[axis] < mySolutionGrids[0].getVoxelRes()[axis] &&
		    rhs.getVoxelRes()[axis] < mySolutionGrids[0].getVoxelRes()[axis]);
	}

	assert(solution.getVoxelRes() == rhs.getVoxelRes());

	mySolutionGrids[0].constant(0);
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

	// Down-stroke of the v-cycle
	for (int level = 0; level < myMGLevels - 1; ++level)
	{
	    // TODO: optimize using method from McAdams et al. 2010 that skips the down-stroke smoothing when using a zero initial guess

	    // Apply smoother
	    for (int smoothIteration = 0; smoothIteration < myTotalSmootherIterations; ++smoothIteration)
	    {
		if (level == 0 && myDoApplyGradientWeights)
		{
		    if (myUseGaussSeidel)
		    {
			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myFineGradientWeights,
							myDx[level],
							true /*smooth odd tiles*/, true /*smooth forwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myFineGradientWeights,
							myDx[level],
							false /*smooth even tiles*/, true /*smooth forwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myFineGradientWeights,
							myDx[level],
							true /*smooth odd tiles*/, false /*smooth backwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myFineGradientWeights,
							myDx[level],
							false /*smooth even tiles*/, false /*smooth backwards*/);
		    }
		    else
		    {
			// TODO: use tiled RB gauss seidel
			dampedJacobiPoissonSmoother(mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myFineGradientWeights,
						    myDx[level]);
		    }

		    // Smooth along boundaries
		    for (int boundaryIteration = 0; boundaryIteration < 2 * (level + 1) * myBoundarySmootherIterations; ++boundaryIteration)
		    {
			dampedJacobiPoissonSmoother(mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myBoundaryCells[level],
						    myFineGradientWeights,
						    myDx[level]);
		    }
		}
		else
		{
		    if (myUseGaussSeidel)
		    {
			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							true /*smooth odd tiles*/, true /*smooth forwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							false /*smooth even tiles*/, true /*smooth forwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							true /*smooth odd tiles*/, false /*smooth backwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							false /*smooth even tiles*/, false /*smooth backwards*/);
		    }
		    else
		    {
			dampedJacobiPoissonSmoother(mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myDx[level]);
		    }

		    for (int boundaryIteration = 0; boundaryIteration < 2 * (level + 1) * myBoundarySmootherIterations; ++boundaryIteration)
		    {
			dampedJacobiPoissonSmoother(mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myBoundaryCells[level],
						    myDx[level]);
		    }
		}
	    }

	    // Compute residual to restrict to the next level
	    if (level == 0 && myDoApplyGradientWeights)
	    {
		computePoissonResidual(myResidualGrids[level],
					mySolutionGrids[level],
					myRHSGrids[level],
					myDomainCellLabels[level],
					myFineGradientWeights,
					myDx[level]);
	    }
	    else
	    {
		computePoissonResidual(myResidualGrids[level],
					mySolutionGrids[level],
					myRHSGrids[level],
					myDomainCellLabels[level],
					myDx[level]);
	    }

	    downsample(myRHSGrids[level + 1],
			myResidualGrids[level],
			myDomainCellLabels[level + 1],
			myDomainCellLabels[level]);

	    mySolutionGrids[level + 1].constant(0);
	}

	copyGridToVector(myCoarseRHSVector,
			    myRHSGrids[myMGLevels - 1],
			    myDirectSolverIndices,
			    myDomainCellLabels[myMGLevels - 1]);

	Vector directSolution = myCoarseSolver.solve(myCoarseRHSVector);

	copyVectorToGrid(mySolutionGrids[myMGLevels - 1],
			    directSolution,
			    myDirectSolverIndices,
			    myDomainCellLabels[myMGLevels - 1]);

	// Up-stroke of the v-cycle
	for (int level = myMGLevels - 2; level >= 0; --level)
	{
	    upsampleAndAdd(mySolutionGrids[level],
			    mySolutionGrids[level + 1],
			    myDomainCellLabels[level],
			    myDomainCellLabels[level + 1]);
	    
	    // Apply smoother
	    for (int smoothIteration = 0; smoothIteration < myTotalSmootherIterations; ++smoothIteration)
	    {
		if (level == 0 && myDoApplyGradientWeights)
		{
		    // Smooth along boundaries
		    for (int boundaryIteration = 0; boundaryIteration < 2 * (level + 1) * myBoundarySmootherIterations; ++boundaryIteration)
		    {
			dampedJacobiPoissonSmoother(mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myBoundaryCells[level],
						    myFineGradientWeights,
						    myDx[level]);
		    }

		    if (myUseGaussSeidel)
		    {
			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myFineGradientWeights,
							myDx[level],
							false /*smooth even tiles*/, true /*smooth forwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myFineGradientWeights,
							myDx[level],
							true /*smooth odd tiles*/, true /*smooth forwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myFineGradientWeights,
							myDx[level],
							false /*smooth even tiles*/, false /*smooth backwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myFineGradientWeights,
							myDx[level],
							true /*smooth odd tiles*/, false /*smooth backwards*/);
		    }
		    else
		    {
			dampedJacobiPoissonSmoother(mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myFineGradientWeights,
						    myDx[level]);
		    }
		}
		else
		{
		    for (int boundaryIteration = 0; boundaryIteration < 2 * (level + 1) * myBoundarySmootherIterations; ++boundaryIteration)
		    {
			dampedJacobiPoissonSmoother(mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myBoundaryCells[level],
						    myDx[level]);
		    }

		    if (myUseGaussSeidel)
		    {
			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							false /*smooth even tiles*/, true /*smooth forwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							true /*smooth odd tiles*/, true /*smooth forwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							false /*smooth even tiles*/, false /*smooth backwards*/);

			tiledGaussSeidelPoissonSmoother(mySolutionGrids[level],
							myRHSGrids[level],
							myDomainCellLabels[level],
							myDx[level],
							true /*smooth odd tiles*/, false /*smooth backwards*/);
		    }
		    else
		    {
			dampedJacobiPoissonSmoother(mySolutionGrids[level],
						    myRHSGrids[level],
						    myDomainCellLabels[level],
						    myDx[level]);
		    }
		}
	    }
	}

	// Copy local solution vector with expanded exterior band to 
	// an interior solution that matches the supplied RHS vector grid.
	copyFromExpandedGrid(solution,
				mySolutionGrids[0],
				myDomainCellLabels[0],
				myExteriorOffset);
    }
}