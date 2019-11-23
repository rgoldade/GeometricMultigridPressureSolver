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

	// TODO - optimize by uncompressing destination tiles that could potentially have cells that are non-exterior

	UTparallelForEachNumber(destinationCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&destinationCellLabels);

	    UT_VoxelProbe<int, true /* read */, false /* no write */, false> fineCellProbe[2][2];

	    for (int zOffset : {0,1})
		for (int yOffset : {0,1})
		    fineCellProbe[yOffset][zOffset].setConstArray(&sourceCellLabels, 0, 1);

	    UT_VoxelProbe<int, false /* no read */, true /* write */, true /* test for write */> destinationCellLabelProbe;
	    destinationCellLabelProbe.setArray(&destinationCellLabels);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		for (; !vit.atEnd(); vit.advance())
		{
		    UT_Vector3I cell(vit.x(), vit.y(), vit.z());

		    bool hasDirichletChild = false;
		    bool hasInteriorChild = false;

		    UT_Vector3I fineCell = cell * 2;

		    for (int zOffset : {0,1})
			for (int yOffset : {0,1})
			{
			    fineCellProbe[yOffset][zOffset].setIndex(fineCell[0], fineCell[1] + yOffset, fineCell[2] + zOffset);

			    for (int xOffset : {0,1})
			    {
				auto label = fineCellProbe[yOffset][zOffset].getValue(xOffset);
				if (label == CellLabels::DIRICHLET_CELL)
				{
				    hasDirichletChild = true;
				}
				else if (label == CellLabels::INTERIOR_CELL || 
					    label == CellLabels::BOUNDARY_CELL)
				    hasInteriorChild = true;
			    }
			}

		    destinationCellLabelProbe.setIndex(vit);

		    if (hasDirichletChild)
			destinationCellLabelProbe.setValue(CellLabels::DIRICHLET_CELL);
		    else if (hasInteriorChild)
			destinationCellLabelProbe.setValue(CellLabels::INTERIOR_CELL);
		    else
			destinationCellLabelProbe.setValue(CellLabels::EXTERIOR_CELL);
		}
	    }
	});

	// Set boundary cells
	UTparallelForEachNumber(destinationCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setArray(&destinationCellLabels);

	    UT_VoxelProbeCube<int> readCellLabelProbe;
	    readCellLabelProbe.setConstPlusArray(&destinationCellLabels);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL)
			{
			    readCellLabelProbe.setIndexPlus(vit);

			    bool hasBoundary = false;
			    for (int axis = 0; axis < 3 && !hasBoundary; ++axis)
				for (int direction : {0,1})
				{
#if !defined(NDEBUG)
				    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
				    UT_Vector3I adjacentCell = SIM::FieldUtils::cellToCellMap(cell, axis, direction);

				    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < destinationCellLabels.getVoxelRes()[axis]);
#endif
				    UT_Vector3I offset(0,0,0);
				    offset[axis] += direction == 0 ? -1 : 1;
				    
				    auto localCellLabel = readCellLabelProbe.getValue(offset[0], offset[1], offset[2]);
				    if (localCellLabel == CellLabels::EXTERIOR_CELL ||
					localCellLabel == CellLabels::DIRICHLET_CELL)
				    {
					hasBoundary = true;
					break;
				    }
				}

			    if (hasBoundary)
				vit.setValue(CellLabels::BOUNDARY_CELL);
			}
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

		if (!vit.isTileConstant() || vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			if (vitt.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
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

		    assert(sourceCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			    sourceCellLabels(cell) == CellLabels::BOUNDARY_CELL);
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
		    assert(sourceCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			    sourceCellLabels(cell) == CellLabels::BOUNDARY_CELL);

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
			assert(sourceCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				sourceCellLabels(cell) == CellLabels::BOUNDARY_CELL);

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
			    
			    assert(sourceCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				    sourceCellLabels(cell) == CellLabels::BOUNDARY_CELL);

			    localBoundaryCellList.append(cell);
			}
#if !defined(NDEBUG)
			else
			{

			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
			    assert(sourceCellLabels(cell) != CellLabels::BOUNDARY_CELL);

			}
#endif
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

	// Sort boundary cells to collect cells within a tile and then along the same axis.
	auto boundaryCellCompare = [&sourceCellLabels](const UT_Vector3I &cellA, const UT_Vector3I &cellB)
	{
	    // Compare tile number first
	    int tileNumberA = sourceCellLabels.indexToLinearTile(cellA[0], cellA[1], cellA[2]);
	    int tileNumberB = sourceCellLabels.indexToLinearTile(cellB[0], cellB[1], cellB[2]);

	    if (tileNumberA < tileNumberB)
		return true;
	    else if (tileNumberA == tileNumberB)
	    {
		if (cellA[2] < cellB[2])
		    return true;
		else if (cellA[2] == cellB[2])
		{
		    if (cellA[1] < cellB[1])
			return true;
		    else if (cellA[1] == cellB[1] &&
			    cellA[0] < cellB[0])
			return true;
		}
	    }

	    return false;
	};

	UTparallelSort(fullBoundaryCellList.begin(), fullBoundaryCellList.end(), boundaryCellCompare);

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

		for (int i = range.begin(); i != range.end(); ++i)
		{
		    vit.myTileStart = i;
		    vit.myTileEnd = i + 1;
		    vit.rewind();

		    if (boss->opInterrupt())
			return;

		    for (; !vit.atEnd(); vit.advance())
		    {
			if (!dirichletTestPassed)
			    return;

			UT_Vector3I fineCell(vit.x(), vit.y(), vit.z());
			UT_Vector3I coarseCell = fineCell / 2;

			// If the fine cell is Dirichlet, it's coarse cell equivalent has to also be Dirichlet
			if (vit.getValue() == CellLabels::DIRICHLET_CELL)
			{
			    if (coarseCellLabels(coarseCell) != CellLabels::DIRICHLET_CELL)
				dirichletTestPassed = false;
			}
			else if (vit.getValue() == CellLabels::INTERIOR_CELL ||
				    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    // If the fine cell is interior, the coarse cell can be either
			    // interior or Dirichlet (if a sibling cell is Dirichlet).
			    if (coarseCellLabels(coarseCell) == CellLabels::EXTERIOR_CELL)
				dirichletTestPassed = false;
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

		for (int i = range.begin(); i != range.end(); ++i)
		{
		    vit.myTileStart = i;
		    vit.myTileEnd = i + 1;
		    vit.rewind();

		    if (boss->opInterrupt())
			return;
		    
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (!coarseningStrategyTestPassed)
			return;

			bool foundDirichletChild = false;
			bool foundInteriorChild = false;
			bool foundExteriorChild = false;

			UT_Vector3I coarseCell(vit.x(), vit.y(), vit.z());

			for (int childCellIndex = 0; childCellIndex < 8; ++childCellIndex)
			{
			    UT_Vector3I fineCell = getChildCell(coarseCell, childCellIndex);

			    auto fineLabel = fineCellLabels(fineCell);

			    if (fineLabel == CellLabels::DIRICHLET_CELL)
				foundDirichletChild = true;
			    else if (fineLabel == CellLabels::INTERIOR_CELL ||
					fineLabel == CellLabels::BOUNDARY_CELL)
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
			else if (coarseLabel == CellLabels::INTERIOR_CELL ||
				    coarseLabel == CellLabels::BOUNDARY_CELL)
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
	    });

	    if (!coarseningStrategyTestPassed) return false;
	}

	return true;
    }

    bool
    unitTestExteriorCells(const UT_VoxelArray<int> &cellLabels)
    {
	const UT_Vector3I voxelRes = cellLabels.getVoxelRes();

	UT_Vector3I startCell(0,0,0);
	UT_Vector3I endCell = voxelRes;

	bool exteriorCellTestPassed = true;
	for (int axis : {0,1,2})
	    for (int direction : {0,1})
	    {
		UT_Vector3I localStartCell = startCell;
		UT_Vector3I localEndCell = endCell;

		if (direction == 0)
		    localEndCell[axis] = 1;
		else
		    localStartCell[axis] = endCell[axis] - 1;

		forEachVoxelRange(localStartCell, localEndCell, [&](const UT_Vector3I &cell)
		{
		    if (!exteriorCellTestPassed) return;

		    if (cellLabels(cell) != CellLabels::EXTERIOR_CELL)
			exteriorCellTestPassed = false;
		});
	    }

	return exteriorCellTestPassed;
    }
} //namespace HDK::GeometricMultigridOperators