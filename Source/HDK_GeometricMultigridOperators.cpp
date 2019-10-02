#include "HDK_GeometricMultigridOperators.h"

using namespace SIM::FieldUtils;

namespace HDK::GeometricMultigridOperators {
    
    SYS_FORCE_INLINE
    UT_Vector3I
    getChildCell(const UT_Vector3I parentCell, int childIndex)
    {
	assert(childIndex < 8);

	UT_Vector3I childCell = 2 * parentCell;
	for (int axis : {0, 1, 2})
	{
	    if (childIndex & (1 << axis))
		++childCell[axis];
	}

	return childCell;
    }

    UT_VoxelArray<int>
    buildCoarseCellLabels(const UT_VoxelArray<int> &sourceCellLabels)
    {

#if !defined(NDEBUG)
	for (int axis : {0,1,2})
	    assert(sourceCellLabels.getVoxelRes()[axis] % 2 == 0);
#endif

	UT_VoxelArray<int> destinationCellLabels;
	destinationCellLabels.size(sourceCellLabels.getVoxelRes()[0] / 2,
				    sourceCellLabels.getVoxelRes()[1] / 2,
				    sourceCellLabels.getVoxelRes()[2] / 2);
	destinationCellLabels.constant(CellLabels::EXTERIOR_CELL);

	UT_Interrupt *boss = UTgetInterrupt();

	// TODO - optimize be uncompressing destination tiles that could potentially have cells that are non-exterior

	UTparallelForEachNumber(destinationCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&destinationCellLabels);
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
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			bool hasDirichletChild = false;
			bool hasInteriorChild = false;
			
			// Iterate over the destination cell's children.
			for (int childCellIndex = 0; childCellIndex < 8; ++childCellIndex)
			{
			    UT_Vector3I childCell = getChildCell(cell, childCellIndex);

			    if (sourceCellLabels(childCell) == CellLabels::DIRICHLET_CELL)
			    {
				hasDirichletChild = true;
				break;
			    }
			    else if (sourceCellLabels(childCell) == CellLabels::INTERIOR_CELL)
				hasInteriorChild = true;
			}

			if (hasDirichletChild)
			    destinationCellLabels.setValue(cell, CellLabels::DIRICHLET_CELL);
			else if (hasInteriorChild)
			    destinationCellLabels.setValue(cell, CellLabels::INTERIOR_CELL);
			else
			    destinationCellLabels.setValue(cell, CellLabels::EXTERIOR_CELL);
		    }
		}
	    }
	});

	destinationCellLabels.collapseAllTiles();
	return destinationCellLabels;
    }

    UT_Array<UT_Vector3I>
    buildBoundaryCells(const UT_VoxelArray<int> &sourceCellLabels,
			const int boundaryWidth)
    {

#if !defined(NDEBUG)
	for (int axis : {0,1,2})
	    assert(sourceCellLabels.getVoxelRes()[axis] % 2 == 0);
#endif

	constexpr int UNVISITED_CELL = 0;
	constexpr int VISITED_CELL = 1;

	UT_VoxelArray<int> visitedCells;
	visitedCells.size(sourceCellLabels.getVoxelRes()[0],
			    sourceCellLabels.getVoxelRes()[1],
			    sourceCellLabels.getVoxelRes()[2]);
	visitedCells.constant(UNVISITED_CELL);

	const int threadCount = UT_Thread::getNumProcessors();
	UT_Array<UT_Array<UT_Vector3I>> parallelBoundaryCellList;
	parallelBoundaryCellList.setSize(threadCount);

	UT_Interrupt *boss = UTgetInterrupt();

	UT_ThreadedAlgorithm buildInitialLayerAlgorithm;
	buildInitialLayerAlgorithm.run([&](const UT_JobInfo &info)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&sourceCellLabels);
	    vit.splitByTile(info);

	    UT_VoxelTileIterator<int> vitt;

	    UT_Array<UT_Vector3I> &localBoundaryCellList = parallelBoundaryCellList[info.job()];

	    for (vit.rewind(); !vit.atEnd(); vit.advanceTile())
	    {
		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
			    
			    bool isBoundaryCell = false;

			    for (int axis = 0; axis < 3 && !isBoundaryCell; ++axis)
				for (int direction : {0, 1})
				{
				    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < sourceCellLabels.getVoxelRes()[axis]);

				    if (sourceCellLabels(adjacentCell) != INTERIOR_CELL)
				    {
					isBoundaryCell = true;
					break;
				    }
				}

			    if (isBoundaryCell)
				localBoundaryCellList.append(cell);
			}
		    }
		}
	    }

	    return 0;
	});

	UT_Array<UT_Vector3I> currentBoundaryCellLayer;
	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(visitedCells.numTiles());

	for (int layer = 0; layer < boundaryWidth; ++layer)
	{
	    // Collect parallel boundary list
	    exint listSize = 0;
	    for (int thread = 0; thread < threadCount; ++thread)
		listSize += parallelBoundaryCellList[thread].size();

	    currentBoundaryCellLayer.clear();
	    currentBoundaryCellLayer.bumpCapacity(listSize);

	    for (int thread = 0; thread < threadCount; ++thread)
	    {
		currentBoundaryCellLayer.concat(parallelBoundaryCellList[thread]);
		parallelBoundaryCellList[thread].clear();
	    }

	    //
	    // Set new layer of boundary cells to visited
	    //

	    isTileOccupiedList.constant(false);

	    // Uncompress tiles in visited grid
	    UTparallelForLightItems(UT_BlockedRange<exint>(0, currentBoundaryCellLayer.size()), [&](const UT_BlockedRange<exint> &range)
	    {
		const int tileCount = visitedCells.numTiles();
		UT_Array<bool> localIsTileOccupiedList;
		localIsTileOccupiedList.setSize(tileCount);
		localIsTileOccupiedList.constant(false);

		if (boss->opInterrupt())
		    return;

		for (exint i = range.begin(); i != range.end(); ++i)
		{
		    if (!(i & 127))
		    {
			if (boss->opInterrupt())
			    return;
		    }

		    UT_Vector3I cell = currentBoundaryCellLayer[i];

		    assert(sourceCellLabels(cell) == CellLabels::INTERIOR_CELL);
		    assert(visitedCells(cell) == UNVISITED_CELL);

		    int tileNumber = visitedCells.indexToLinearTile(cell[0], cell[1], cell[2]);
		    assert(sourceCellLabels.indexToLinearTile(cell[0], cell[1], cell[2]) == tileNumber);

		    if (!localIsTileOccupiedList[tileNumber])
			localIsTileOccupiedList[tileNumber] = true;
		}

		for (int tileNumber = 0; tileNumber < tileCount; ++tileNumber)
		{
		    if (localIsTileOccupiedList[tileNumber] && !isTileOccupiedList[tileNumber])
			isTileOccupiedList[tileNumber] = true;
		}
	    });

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
			visitedCells.getLinearTile(i)->uncompress();
		}
	    });

	    // Set visited cells
	    UTparallelForLightItems(UT_BlockedRange<exint>(0, currentBoundaryCellLayer.size()), [&](const UT_BlockedRange<exint> &range)
	    {
		if (boss->opInterrupt())
		    return;

		for (exint i = range.begin(); i != range.end(); ++i)
		{
		    if (!(i & 127))
		    {
			if (boss->opInterrupt())
			    return;
		    }

		    UT_Vector3I cell = currentBoundaryCellLayer[i];
		    
		    // Verify that the tile has been uncompressed
		    assert(!visitedCells.getLinearTile(visitedCells.indexToLinearTile(cell[0], cell[1], cell[2]))->isConstant());
		    assert(sourceCellLabels(cell) == CellLabels::INTERIOR_CELL);

		    visitedCells.setValue(cell, VISITED_CELL);
		}
	    });

	    // Collect new layer of boundary cells
	    if (layer < boundaryWidth - 1)
	    {
		UT_ThreadedAlgorithm buildNextLayerAlgorithm;
		buildNextLayerAlgorithm.run([&](const UT_JobInfo &info)
		{
		    exint start, end;
		    const exint elementSize = currentBoundaryCellLayer.entries();
		    info.divideWork(elementSize, start, end);

		    if (boss->opInterrupt())
			return 0;

		    UT_Array<UT_Vector3I> &localNewBoundaryCellList = parallelBoundaryCellList[info.job()];
		    const exint localEnd = end;
		    for (exint i = start; i < localEnd; ++i)
		    {
			if (!(i & 127))
			{
			    if (boss->opInterrupt())
				break;
			}

			UT_Vector3I cell = currentBoundaryCellLayer[i];

			assert(visitedCells(cell) == VISITED_CELL);
			assert(sourceCellLabels(cell) == CellLabels::INTERIOR_CELL);

			// Load up neighbouring INTERIOR cells that have not been visited
			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < sourceCellLabels.getVoxelRes()[axis]);

				if (sourceCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL &&
				    visitedCells(adjacentCell) == UNVISITED_CELL)
				    localNewBoundaryCellList.append(adjacentCell);

			    }
		    }

		    return 0;
		});
	    }
	}

	// Compile final list of boundary cells
	UT_ThreadedAlgorithm buildFinalLayerAlgorithm;
	buildFinalLayerAlgorithm.run([&](const UT_JobInfo &info)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&visitedCells);
	    vit.splitByTile(info);

	    UT_VoxelTileIterator<int> vitt;

	    UT_Array<UT_Vector3I> &localBoundaryCellList = parallelBoundaryCellList[info.job()];

	    for (vit.rewind(); !vit.atEnd(); vit.advanceTile())
	    {
		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() || vit.getValue() == VISITED_CELL)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			if (vitt.getValue() == VISITED_CELL)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
			    assert(sourceCellLabels(cell) == CellLabels::INTERIOR_CELL);
			    localBoundaryCellList.append(cell);
			}
		    }
		}
	    }

	    return 0;
	});


	// Collect parallel boundary list
	exint listSize = 0;
	for (int thread = 0; thread < threadCount; ++thread)
	    listSize += parallelBoundaryCellList[thread].size();

	UT_Array<UT_Vector3I> fullBoundaryCellList;
	fullBoundaryCellList.bumpCapacity(listSize);

	for (int thread = 0; thread < threadCount; ++thread)
	    fullBoundaryCellList.concat(parallelBoundaryCellList[thread]);

	return fullBoundaryCellList;
    }

    bool
    unitTestCoarsening(const UT_VoxelArray<int> &coarseCellLabels,
			const UT_VoxelArray<int> &fineCellLabels)
    {
	// The coarse cell grid must be exactly have the size of the fine cell grid.
	if (2 * coarseCellLabels.getVoxelRes() != fineCellLabels.getVoxelRes())
	    return false;

	UT_Interrupt *boss = UTgetInterrupt();

	for (int axis : {0,1,2})
	{
	    if (!(coarseCellLabels.getVoxelRes()[axis] % 2 == 0 &&
		    fineCellLabels.getVoxelRes()[axis] % 2 == 0)) return false;
	}

	{
	    bool dirichletTestPassed = true;

	    UTparallelForEachNumber(fineCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&fineCellLabels);
		UT_VoxelTileIterator<int> vitt;

		for (int i = range.begin(); i != range.end(); ++i)
		{
		    vit.myTileStart = i;
		    vit.myTileEnd = i + 1;
		    vit.rewind();

		    if (boss->opInterrupt())
			return;

		    if (!vit.atEnd())
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (!dirichletTestPassed)
				return;

			    UT_Vector3I fineCell(vitt.x(), vitt.y(), vitt.z());
			    UT_Vector3I coarseCell = fineCell / 2;
			    // If the fine cell is Dirichlet, it's coarse cell equivalent has to also be Dirichlet
			    if (vitt.getValue() == CellLabels::DIRICHLET_CELL)
			    {
				if (coarseCellLabels(coarseCell) != CellLabels::DIRICHLET_CELL)
				    dirichletTestPassed = false;
			    }
			    else if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				// If the fine cell is interior, the coarse cell can be either
				// interior or Dirichlet (if a sibling cell is Dirichlet).
				if (coarseCellLabels(coarseCell) == CellLabels::EXTERIOR_CELL)
				    dirichletTestPassed = false;
			    }
			}
		    }
		}
	    });

	    if (!dirichletTestPassed) return false;
	}
	{
	    bool coarseningStrategyTestPassed = true;

	    UTparallelForEachNumber(coarseCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&coarseCellLabels);
		UT_VoxelTileIterator<int> vitt;

		for (int i = range.begin(); i != range.end(); ++i)
		{
		    vit.myTileStart = i;
		    vit.myTileEnd = i + 1;
		    vit.rewind();

		    if (boss->opInterrupt())
			return;

		    if (!vit.atEnd())
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (!coarseningStrategyTestPassed)
				return;

			    bool foundDirichletChild = false;
			    bool foundInteriorChild = false;
			    bool foundExteriorChild = false;

			    UT_Vector3I coarseCell(vitt.x(), vitt.y(), vitt.z());

			    for (int childCellIndex = 0; childCellIndex < 8; ++childCellIndex)
			    {
				UT_Vector3I fineCell = getChildCell(coarseCell, childCellIndex);

				auto fineLabel = fineCellLabels(fineCell);

				if (fineLabel == CellLabels::DIRICHLET_CELL)
				    foundDirichletChild = true;
				else if (fineLabel == CellLabels::INTERIOR_CELL)
				    foundInteriorChild = true;
				else if (fineLabel == CellLabels::EXTERIOR_CELL)
				    foundExteriorChild = true;
			    }

			    auto coarseLabel = coarseCellLabels(coarseCell);
			    if (coarseLabel == CellLabels::DIRICHLET_CELL)
			    {
				if (!foundDirichletChild)
				    coarseningStrategyTestPassed = false;
			    }
			    else if (coarseLabel == CellLabels::INTERIOR_CELL)
			    {
				if (foundDirichletChild || !foundInteriorChild)
				    coarseningStrategyTestPassed = false;
			    }
			    else if (coarseLabel == CellLabels::EXTERIOR_CELL)
			    {
				if (foundDirichletChild || foundInteriorChild || !foundExteriorChild)
				    coarseningStrategyTestPassed = false;
			    }
			}
		    }
		}
	    });

	    if (!coarseningStrategyTestPassed) return false;
	}

	return true;
    }    
} //namespace HDK::GeometricMultigridOperators