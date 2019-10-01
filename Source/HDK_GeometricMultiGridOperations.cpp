#include "HDK_GeometricMultiGridOperations.h"

#include <SIM/SIM_FieldUtils.h>
#include <UT/UT_ParallelUtil.h>

using namespace SIM::FieldUtils;

namespace HDK::GeometricMultiGridOperations {

    void
    dampedJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				const UT_VoxelArray<StoreReal> &rhs,
				const UT_VoxelArray<int> &cellLabels,
				const SolveReal dx)
    {
	const UT_Vector3I voxelRes = solution.getVoxelRes();

	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	UT_VoxelArray<StoreReal> tempSolution = solution;

	// TODO: factor dx terms out
	SolveReal gridScalar = 1. / (dx * dx);

	UT_Interrupt *boss = UTgetInterrupt();

        UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				SolveReal laplacian = 0;
				SolveReal count = 0;

				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

					// Geometric multigrid uses padding at the boundary limits so
					// we should never end up out of bounds.
					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

					if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
					{
					    laplacian -= tempSolution(adjacentCell);
					    ++count;
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
					    ++count;
				    }

				laplacian += count * SolveReal(tempSolution(cell));
				laplacian *= gridScalar;
				SolveReal residual = SolveReal(rhs(cell)) - laplacian;
				residual /= (count * gridScalar);

				solution.setValue(cell, solution(cell) + SolveReal(2. / 3.) * residual);
			    }
			}
		    }
		}
	    }
	});
    }

    void
    dampedJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				const UT_VoxelArray<StoreReal> &rhs,
				const UT_VoxelArray<int> &cellLabels,
				const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
				const SolveReal dx)
    {
	const UT_Vector3I voxelRes = solution.getVoxelRes();

	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	assert( gradientWeights[0].getVoxelRes()[0] - 1 == solution.getVoxelRes()[0] &&
		gradientWeights[0].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		gradientWeights[0].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		gradientWeights[1].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		gradientWeights[1].getVoxelRes()[1] - 1 == solution.getVoxelRes()[1] &&
		gradientWeights[1].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		gradientWeights[2].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		gradientWeights[2].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		gradientWeights[2].getVoxelRes()[2] - 1 == solution.getVoxelRes()[2]);

	SolveReal gridScalar = 1. / (dx * dx);

	UT_VoxelArray<StoreReal> tempSolution = solution;

	UT_Interrupt *boss = UTgetInterrupt();

        UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				SolveReal laplacian = 0;
				SolveReal diagonal = 0;

				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);
					UT_Vector3I face = cellToFaceMap(cell, axis, direction);
					
					// Geometric multigrid uses padding at the boundary limits so
					// we should never end up out of bounds.
					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

					if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
					{
					    laplacian -= SolveReal(gradientWeights[axis](face)) * SolveReal(tempSolution(adjacentCell));
					    diagonal += gradientWeights[axis](face);
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
					    diagonal += gradientWeights[axis](face);
				    }

				laplacian += diagonal * SolveReal(tempSolution(cell));
				laplacian *= gridScalar;
				SolveReal residual = SolveReal(rhs(cell)) - laplacian;
				residual /= (diagonal * gridScalar);

				solution.setValue(cell, solution(cell) + SolveReal(2. / 3.) * residual);
			    }
			}
		    }
		}
	    }
	});
    }

    void
    dampedJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				const UT_VoxelArray<StoreReal> &rhs,
				const UT_VoxelArray<int> &cellLabels,
				const UT_Array<UT_Vector3I> &boundaryCells,
				const SolveReal dx)
    {
	const UT_Vector3I voxelRes = solution.getVoxelRes();

	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	const exint listSize = boundaryCells.size();

	SolveReal gridScalar = 1. / (dx * dx);

	UT_Array<StoreReal> tempSolution;
	tempSolution.setSize(listSize);

	UT_Interrupt *boss = UTgetInterrupt();

	// Apply Jacobi smoothing for boundary cell items and store in a temporary list
	UTparallelForLightItems(UT_BlockedRange<exint>(0, listSize), [&](const UT_BlockedRange<exint> &range)
	{
	    if (boss->opInterrupt())
		return;

	    for (exint cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
	    {
		UT_Vector3I cell = boundaryCells[cellIndex];

		assert(cellLabels(cell) == CellLabels::INTERIOR_CELL);

		SolveReal laplacian = 0;
		SolveReal count = 0;

		for (int axis : {0,1,2})
		    for (int direction : {0,1})
		    {
			UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

			// Geometric multigrid uses padding at the boundary limits so
			// we should never end up out of bounds.
			assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

			if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
			{
			    laplacian -= solution(adjacentCell);
			    ++count;
			}
			else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
			    ++count;
		    }

		laplacian += count * SolveReal(solution(cell));
		laplacian *= gridScalar;
		SolveReal residual = SolveReal(rhs(cell)) - laplacian;
		residual /= (count * gridScalar);

		tempSolution[cellIndex] = SolveReal(solution(cell)) +  SolveReal(2. / 3.) * residual;
	    }
	});

	// Apply updated solution in the temporary array to the solution grid
	UTparallelForLightItems(UT_BlockedRange<exint>(0, listSize), [&](const UT_BlockedRange<exint> &range)
	{
	    if (boss->opInterrupt())
		return;

	    for (exint cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
	    {
		UT_Vector3I cell = boundaryCells[cellIndex];

		// The tile that the cell falls into in the solution grid MUST be uncompressed
		assert(!solution.getLinearTile(solution.indexToLinearTile(cell[0], cell[1], cell[2]))->isConstant());

		solution.setValue(cell, tempSolution[cellIndex]);
	    }
	});
    }

    void
    dampedJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				const UT_VoxelArray<StoreReal> &rhs,
				const UT_VoxelArray<int> &cellLabels,
				const UT_Array<UT_Vector3I> &boundaryCells,
				const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
				const SolveReal dx)
    {
	const UT_Vector3I voxelRes = solution.getVoxelRes();

	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	assert( gradientWeights[0].getVoxelRes()[0] - 1 == solution.getVoxelRes()[0] &&
		gradientWeights[0].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		gradientWeights[0].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		gradientWeights[1].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		gradientWeights[1].getVoxelRes()[1] - 1 == solution.getVoxelRes()[1] &&
		gradientWeights[1].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		gradientWeights[2].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		gradientWeights[2].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		gradientWeights[2].getVoxelRes()[2] - 1 == solution.getVoxelRes()[2]);

	const exint listSize = boundaryCells.size();

	SolveReal gridScalar = 1. / (dx * dx);

	UT_Array<StoreReal> tempSolution;
	tempSolution.setSize(listSize);

	UT_Interrupt *boss = UTgetInterrupt();

	// Apply Jacobi smoothing for boundary cell items and store in a temporary list
	UTparallelForLightItems(UT_BlockedRange<exint>(0, listSize), [&](const UT_BlockedRange<exint> &range)
	{
	    if (boss->opInterrupt())
		return;

	    for (exint cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
	    {
		UT_Vector3I cell = boundaryCells[cellIndex];

		assert(cellLabels(cell) == CellLabels::INTERIOR_CELL);

		SolveReal laplacian = 0;
		SolveReal diagonal = 0;

		for (int axis : {0,1,2})
		    for (int direction : {0,1})
		    {
			UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);
			UT_Vector3I face = cellToFaceMap(cell, axis, direction);
			
			// Geometric multigrid uses padding at the boundary limits so
			// we should never end up out of bounds.
			assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

			if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
			{
			    laplacian -= SolveReal(gradientWeights[axis](face)) * SolveReal(solution(adjacentCell));
			    diagonal += gradientWeights[axis](face);
			}
			else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
			    diagonal += gradientWeights[axis](face);
		    }

		laplacian += diagonal * SolveReal(solution(cell));
		laplacian *= gridScalar;
		SolveReal residual = SolveReal(rhs(cell)) - laplacian;
		residual /= (diagonal * gridScalar);

		tempSolution[cellIndex] = SolveReal(solution(cell)) +  SolveReal(2. / 3.) * residual;
	    }
	});

	// Apply updated solution in the temporary array to the solution grid
	UTparallelForLightItems(UT_BlockedRange<exint>(0, listSize), [&](const UT_BlockedRange<exint> &range)
	{
	    if (boss->opInterrupt())
		return;

	    for (exint cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
	    {
		UT_Vector3I cell = boundaryCells[cellIndex];

		// The tile that the cell falls into in the solution grid MUST be uncompressed
		assert(!solution.getLinearTile(solution.indexToLinearTile(cell[0], cell[1], cell[2]))->isConstant());

		solution.setValue(cell, tempSolution[cellIndex]);
	    }
	});
    }

    void
    tiledGaussSeidelPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const SolveReal dx,
				    const bool doSmoothOddTiles,
				    const bool doSmoothForward)
    {
	const UT_Vector3I voxelRes = solution.getVoxelRes();

	SolveReal gridScalar = 1. / (dx * dx);

	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
			// TODO: This could probably be done more efficiently
			// Filter out odd or even tiles
			int tileNumber = vit.getLinearTileNum();
			UT_Vector3i tileIndex;
			cellLabels.linearTileToXYZ(tileNumber, tileIndex[0], tileIndex[1], tileIndex[2]);

			int oddCount = 0;
			for (int axis : {0,1,2})
			    oddCount += tileIndex[axis];

			bool isTileOdd = (oddCount % 2 != 0);
			// We should only be applying to odd tiles and the count is even so skip this tile
			if ((doSmoothOddTiles && !isTileOdd) ||
			    (!doSmoothOddTiles && isTileOdd))
			    continue;

			UT_Vector3I tileStart, tileEnd;
			vit.getTileVoxels(tileStart,tileEnd);

			auto gsSmoother = [&](const UT_Vector3I& cell)
			{
			    if (cellLabels(cell) == CellLabels::INTERIOR_CELL)
			    {
				// Compute laplacian
				SolveReal laplacian = 0;
				SolveReal count = 0;
				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);
					
					// Geometric multigrid uses padding at the boundary limits so
					// we should never end up out of bounds.
					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);
					
					if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
					{
					    laplacian -= solution(adjacentCell);
					    ++count;
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
					    ++count;
				    }
				// This is redundant but is easier to understand
				laplacian *= gridScalar;
				SolveReal residual = SolveReal(rhs(cell)) - laplacian;
				residual /= (count * gridScalar);

				solution.setValue(cell, residual);
			    }
			};

			if (doSmoothForward)
			{
			    UT_Vector3I cell;
			    for (cell[0] = tileStart[0]; cell[0] < tileEnd[0]; ++cell[0])
				for (cell[1] = tileStart[1]; cell[1] < tileEnd[1]; ++cell[1])
				    for (cell[2] = tileStart[2]; cell[2] < tileEnd[2]; ++cell[2])
				    {
					gsSmoother(cell);
				    }
			}
			else
			{
			    UT_Vector3I cell;
			    for (cell[0] = tileEnd[0] - 1; cell[0] >= tileStart[0]; --cell[0])
				for (cell[1] = tileEnd[1] - 1; cell[1] >= tileStart[1]; --cell[1])
				    for (cell[2] = tileEnd[2] - 1; cell[2] >= tileStart[2]; --cell[2])
				    {
					gsSmoother(cell);
				    }
			}
		    }
		}
	    }
	});
    }

    void
    tiledGaussSeidelPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
				    const SolveReal dx,
				    const bool doSmoothOddTiles,
				    const bool doSmoothForward)
    {
	const UT_Vector3I voxelRes = solution.getVoxelRes();

	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	assert( gradientWeights[0].getVoxelRes()[0] - 1 == solution.getVoxelRes()[0] &&
		gradientWeights[0].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		gradientWeights[0].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		gradientWeights[1].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		gradientWeights[1].getVoxelRes()[1] - 1 == solution.getVoxelRes()[1] &&
		gradientWeights[1].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		gradientWeights[2].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		gradientWeights[2].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		gradientWeights[2].getVoxelRes()[2] - 1 == solution.getVoxelRes()[2]);

	UT_Interrupt *boss = UTgetInterrupt();

	SolveReal gridScalar = 1. / (dx * dx);

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
			// TODO: This could probably be done more efficiently
			// Filter out odd or even tiles
			int tileNumber = vit.getLinearTileNum();
			UT_Vector3i tileIndex;
			cellLabels.linearTileToXYZ(tileNumber, tileIndex[0], tileIndex[1], tileIndex[2]);

			int oddCount = 0;
			for (int axis : {0,1,2})
			    oddCount += tileIndex[axis];

			bool isTileOdd = (oddCount % 2 != 0);
			// We should only be applying to odd tiles and the count is even so skip this tile
			if ((doSmoothOddTiles && !isTileOdd) ||
			    (!doSmoothOddTiles && isTileOdd))
			    continue;

			UT_Vector3I tileStart, tileEnd;
			vit.getTileVoxels(tileStart,tileEnd);

			auto gsSmoother = [&](const UT_Vector3I& cell)
			{
			    if (cellLabels(cell) == CellLabels::INTERIOR_CELL)
			    {
				// Compute laplacian
				SolveReal laplacian = 0;
				SolveReal diagonal = 0;

				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);
					UT_Vector3I face = cellToFaceMap(cell, axis, direction);

					// Geometric multigrid uses padding at the boundary limits so
					// we should never end up out of bounds.
					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);
					
					if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
					{
					    laplacian -= SolveReal(gradientWeights[axis](face)) * SolveReal(solution(adjacentCell));
					    diagonal += gradientWeights[axis](face);
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
					    diagonal += gradientWeights[axis](face);
				    }

				laplacian *= gridScalar;
				SolveReal residual = SolveReal(rhs(cell)) - laplacian;
				residual /= (diagonal * gridScalar);

				solution.setValue(cell, residual);
			    }
			};

			if (doSmoothForward)
			{
			    UT_Vector3I cell;
			    for (cell[0] = tileStart[0]; cell[0] < tileEnd[0]; ++cell[0])
				for (cell[1] = tileStart[1]; cell[1] < tileEnd[1]; ++cell[1])
				    for (cell[2] = tileStart[2]; cell[2] < tileEnd[2]; ++cell[2])
				    {
					gsSmoother(cell);
				    }
			}
			else
			{
			    UT_Vector3I cell;
			    for (cell[0] = tileEnd[0] - 1; cell[0] >= tileStart[0]; --cell[0])
				for (cell[1] = tileEnd[1] - 1; cell[1] >= tileStart[1]; --cell[1])
				    for (cell[2] = tileEnd[2] - 1; cell[2] >= tileStart[2]; --cell[2])
				    {
					gsSmoother(cell);
				    }
			}
		    }
		}
	    }
	});
    }

    void
    applyPoissonMatrix(UT_VoxelArray<StoreReal> &destination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &cellLabels,
			const SolveReal dx)
    {
	const UT_Vector3I voxelRes = destination.getVoxelRes();

	assert(destination.getVoxelRes() == source.getVoxelRes() &&
		source.getVoxelRes() == cellLabels.getVoxelRes());

	SolveReal gridScalar = 1. / (dx * dx);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				SolveReal laplacian = 0;
				SolveReal count = 0;

				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

					// Geometric multigrid uses padding at the boundary limits so
					// we should never end up out of bounds.
					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

					if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
					{
					    laplacian -= source(adjacentCell);
					    ++count;
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
					    ++count;
				    }

				laplacian += count * SolveReal(source(cell));
				laplacian *= gridScalar;
				destination.setValue(cell, laplacian);
			    }
			}
		    }
		}
	    }
	});
    }

    void
    applyPoissonMatrix(UT_VoxelArray<StoreReal> &destination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &cellLabels,
			const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
			const SolveReal dx)
    {
	const UT_Vector3I voxelRes = destination.getVoxelRes();

	assert(destination.getVoxelRes() == source.getVoxelRes() &&
		source.getVoxelRes() == cellLabels.getVoxelRes());

	assert( gradientWeights[0].getVoxelRes()[0] - 1 == source.getVoxelRes()[0] &&
		gradientWeights[0].getVoxelRes()[1]     == source.getVoxelRes()[1] &&
		gradientWeights[0].getVoxelRes()[2]     == source.getVoxelRes()[2] &&

		gradientWeights[1].getVoxelRes()[0]     == source.getVoxelRes()[0] &&
		gradientWeights[1].getVoxelRes()[1] - 1 == source.getVoxelRes()[1] &&
		gradientWeights[1].getVoxelRes()[2]     == source.getVoxelRes()[2] &&

		gradientWeights[2].getVoxelRes()[0]     == source.getVoxelRes()[0] &&
		gradientWeights[2].getVoxelRes()[1]     == source.getVoxelRes()[1] &&
		gradientWeights[2].getVoxelRes()[2] - 1 == source.getVoxelRes()[2]);

	SolveReal gridScalar = 1. / (dx * dx);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				SolveReal laplacian = 0;
				SolveReal diagonal = 0;

				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);
					UT_Vector3I face = cellToFaceMap(cell, axis, direction);

					// Geometric multigrid uses padding at the boundary limits so
					// we should never end up out of bounds.
					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

					if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
					{
					    laplacian -= SolveReal(gradientWeights[axis](face)) * SolveReal(source(adjacentCell));
					    diagonal += gradientWeights[axis](face);
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
					    diagonal += gradientWeights[axis](face);
				    }

				laplacian += diagonal * SolveReal(source(cell));
				laplacian *= gridScalar;
				destination.setValue(cell, laplacian);
			    }
			}
		    }
		}
	    }
	});
    }

    void
    computePoissonResidual(UT_VoxelArray<StoreReal> &residual,
			    const UT_VoxelArray<StoreReal> &solution,
			    const UT_VoxelArray<StoreReal> &rhs,
			    const UT_VoxelArray<int> &cellLabels,
			    const SolveReal dx)
    {
	const UT_Vector3I voxelRes = residual.getVoxelRes();

	assert(residual.getVoxelRes() == solution.getVoxelRes() &&
		solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	SolveReal gridScalar = 1. / (dx * dx);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				SolveReal laplacian = 0;
				SolveReal count = 0;

				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

					// Geometric multigrid uses padding at the boundary limits so
					// we should never end up out of bounds.
					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

					if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
					{
					    laplacian -= solution(adjacentCell);
					    ++count;
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
					    ++count;
				    }

				laplacian += count * SolveReal(solution(cell));
				laplacian *= gridScalar;
				residual.setValue(cell, SolveReal(rhs(cell)) - laplacian);
			    }
			}
		    }
		}
	    }
	});
    }

    void
    computePoissonResidual(UT_VoxelArray<StoreReal> &residual,
			    const UT_VoxelArray<StoreReal> &solution,
			    const UT_VoxelArray<StoreReal> &rhs,
			    const UT_VoxelArray<int> &cellLabels,
			    const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
			    const SolveReal dx)
    {
	const UT_Vector3I voxelRes = residual.getVoxelRes();

	assert(residual.getVoxelRes() == solution.getVoxelRes() &&
		solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	assert( gradientWeights[0].getVoxelRes()[0] - 1 == solution.getVoxelRes()[0] &&
		gradientWeights[0].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		gradientWeights[0].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		gradientWeights[1].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		gradientWeights[1].getVoxelRes()[1] - 1 == solution.getVoxelRes()[1] &&
		gradientWeights[1].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		gradientWeights[2].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		gradientWeights[2].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		gradientWeights[2].getVoxelRes()[2] - 1 == solution.getVoxelRes()[2]);

	SolveReal gridScalar = 1. / (dx * dx);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				SolveReal laplacian = 0;
				SolveReal diagonal = 0;

				for (int axis : {0,1,2})
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);
					UT_Vector3I face = cellToFaceMap(cell, axis, direction);

					// Geometric multigrid uses padding at the boundary limits so
					// we should never end up out of bounds.
					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

					if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
					{
					    laplacian -= SolveReal(gradientWeights[axis](face)) * SolveReal(solution(adjacentCell));
					    diagonal += gradientWeights[axis](face);
					}
					else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
					    diagonal += gradientWeights[axis](face);
				    }

				laplacian += diagonal * SolveReal(solution(cell));
				laplacian *= gridScalar;
				residual.setValue(cell, SolveReal(rhs(cell)) - laplacian);
			    }
			}
		    }
		}
	    }
	});
    }

    void
    downsample(UT_VoxelArray<StoreReal> &destination,
		const UT_VoxelArray<StoreReal> &source,
		const UT_VoxelArray<int> &destinationCellLabels,
		const UT_VoxelArray<int> &sourceCellLabels)
    {
	constexpr SolveReal restrictionWeights[4] = { 1. / 8., 3. / 8., 3. / 8., 1. / 8. };

	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert(2 * destination.getVoxelRes() == source.getVoxelRes());
	assert(destination.getVoxelRes() == destinationCellLabels.getVoxelRes());
	assert(source.getVoxelRes() == sourceCellLabels.getVoxelRes());

	const UT_Vector3I sourceVoxelRes = source.getVoxelRes();

	for (int axis : {0,1,2})
	{
	    assert(destination.getVoxelRes()[axis] % 2 == 0);
	    assert(source.getVoxelRes()[axis] % 2 == 0);
	}

	UT_Interrupt *boss = UTgetInterrupt();

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
		    if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
                        {
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				SolveReal sampleValue = 0;

				UT_Vector3I startCell = 2 * cell - UT_Vector3I(1);
				
				forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(4)), [&](const UT_Vector3I &sampleOffset)
				{
				    UT_Vector3I sampleCell = startCell + sampleOffset;
				    
				    assert(sampleCell[0] >= 0 && sampleCell[0] < sourceVoxelRes[0] &&
					    sampleCell[1] >= 0 && sampleCell[1] < sourceVoxelRes[1] &&
					    sampleCell[2] >= 0 && sampleCell[2] < sourceVoxelRes[2]);

				    if (sourceCellLabels(sampleCell) == CellLabels::INTERIOR_CELL)
					sampleValue += restrictionWeights[sampleOffset[0]] *
							restrictionWeights[sampleOffset[1]] *
							restrictionWeights[sampleOffset[2]] *
							SolveReal(source(sampleCell));

				});

				destination.setValue(cell, sampleValue);
			    }
			}
		    }
		}
	    }
	});
    }

    // !! WARNING !!
    // We roll our own linear interpolation method here because Houdini's
    // SYSlerp/SYSbilerp method will break symmetry in restriction / prolongation

    template<typename SolveReal>
    SYS_FORCE_INLINE
    SolveReal
    lerp(SolveReal value0, SolveReal value1, SolveReal f)
    {
	return (1. - f) * value0 + f * value1;
    }

    template<typename SolveReal>
    SYS_FORCE_INLINE
    SolveReal
    bilerp(SolveReal value00, SolveReal value10,
	    SolveReal value01, SolveReal value11,
	    SolveReal fx, SolveReal fy)
    {
	return lerp(lerp(value00, value10, fx),
		    lerp(value01, value11, fx), fy);
    }

    template<typename SolveReal>
    SYS_FORCE_INLINE
    SolveReal
    trilerp(SolveReal value000, SolveReal value100,
	    SolveReal value010, SolveReal value110,
	    SolveReal value001, SolveReal value101,
	    SolveReal value011, SolveReal value111,
	    SolveReal fx, SolveReal fy, SolveReal fz)
    {
	return lerp(bilerp(value000, value100, value010, value110, fx, fy),
		    bilerp(value001, value101, value011, value111, fx, fy), fz);
    }

    void
    upsample(UT_VoxelArray<StoreReal> &destination,
		const UT_VoxelArray<StoreReal> &source,
		const UT_VoxelArray<int> &destinationCellLabels,
		const UT_VoxelArray<int> &sourceCellLabels)
    {
	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert(destination.getVoxelRes() / 2 == source.getVoxelRes());
	assert(destination.getVoxelRes() == destinationCellLabels.getVoxelRes());
	assert(source.getVoxelRes() == sourceCellLabels.getVoxelRes());

	for (int axis : {0,1,2})
	{
	    assert(destination.getVoxelRes()[axis] % 2 == 0);
	    assert(source.getVoxelRes()[axis] % 2 == 0);
	}

	UT_Interrupt *boss = UTgetInterrupt();

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
		    if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
                        {
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				UT_Vector3T<SolveReal> samplePoint = .5 * (UT_Vector3T<SolveReal>(cell) + UT_Vector3T<SolveReal>(.5)) - UT_Vector3T<SolveReal>(.5);

				UT_Vector3I startCell = UT_Vector3I(samplePoint);

				UT_Vector3T<SolveReal> interpWeight = samplePoint - UT_Vector3T<SolveReal>(startCell);

				// Hard code interpolation
				// TODO: turn into loop to keep things condensed
				SolveReal v000 = (sourceCellLabels(startCell) == CellLabels::INTERIOR_CELL) ? source(startCell) : 0;
				SolveReal v010 = (sourceCellLabels(startCell + UT_Vector3I(0, 1, 0)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(0, 1, 0)) : 0;
				SolveReal v100 = (sourceCellLabels(startCell + UT_Vector3I(1, 0, 0)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(1, 0, 0)) : 0;
				SolveReal v110 = (sourceCellLabels(startCell + UT_Vector3I(1, 1, 0)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(1, 1, 0)) : 0;

				SolveReal v001 = (sourceCellLabels(startCell + UT_Vector3I(0, 0, 1)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(0, 0, 1)) : 0;
				SolveReal v011 = (sourceCellLabels(startCell + UT_Vector3I(0, 1, 1)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(0, 1, 1)) : 0;
				SolveReal v101 = (sourceCellLabels(startCell + UT_Vector3I(1, 0, 1)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(1, 0, 1)) : 0;
				SolveReal v111 = (sourceCellLabels(startCell + UT_Vector3I(1, 1, 1)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(1, 1, 1)) : 0;

				destination.setValue(cell, trilerp(v000, v100, v010, v110,
								    v001, v101, v011, v111,
								    interpWeight[0], interpWeight[1], interpWeight[2]));
			    }
			}
		    }
		}
	    }
	});
    }

    void
    upsampleAndAdd(UT_VoxelArray<StoreReal> &destination,
		    const UT_VoxelArray<StoreReal> &source,
		    const UT_VoxelArray<int> &destinationCellLabels,
		    const UT_VoxelArray<int> &sourceCellLabels)
    {
	// Make sure both source and destination grid are powers of 2 and one level apart.
	assert(destination.getVoxelRes() / 2 == source.getVoxelRes());
	assert(destination.getVoxelRes() == destinationCellLabels.getVoxelRes());
	assert(source.getVoxelRes() == sourceCellLabels.getVoxelRes());

	for (int axis : {0,1,2})
	{
	    assert(destination.getVoxelRes()[axis] % 2 == 0);
	    assert(source.getVoxelRes()[axis] % 2 == 0);
	}

	UT_Interrupt *boss = UTgetInterrupt();

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
		    if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
                        {
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				UT_Vector3T<SolveReal> samplePoint = .5 * (UT_Vector3T<SolveReal>(cell) + UT_Vector3T<SolveReal>(.5)) - UT_Vector3T<SolveReal>(.5);

				UT_Vector3I startCell = UT_Vector3I(samplePoint);

				UT_Vector3T<SolveReal> interpWeight = samplePoint - UT_Vector3T<SolveReal>(startCell);

				// Hard code interpolation
				// TODO: turn into loop to keep things condensed
				SolveReal v000 = (sourceCellLabels(startCell) == CellLabels::INTERIOR_CELL) ? source(startCell) : 0;
				SolveReal v010 = (sourceCellLabels(startCell + UT_Vector3I(0, 1, 0)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(0, 1, 0)) : 0;
				SolveReal v100 = (sourceCellLabels(startCell + UT_Vector3I(1, 0, 0)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(1, 0, 0)) : 0;
				SolveReal v110 = (sourceCellLabels(startCell + UT_Vector3I(1, 1, 0)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(1, 1, 0)) : 0;

				SolveReal v001 = (sourceCellLabels(startCell + UT_Vector3I(0, 0, 1)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(0, 0, 1)) : 0;
				SolveReal v011 = (sourceCellLabels(startCell + UT_Vector3I(0, 1, 1)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(0, 1, 1)) : 0;
				SolveReal v101 = (sourceCellLabels(startCell + UT_Vector3I(1, 0, 1)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(1, 0, 1)) : 0;
				SolveReal v111 = (sourceCellLabels(startCell + UT_Vector3I(1, 1, 1)) == CellLabels::INTERIOR_CELL) ? source(startCell + UT_Vector3I(1, 1, 1)) : 0;

				destination.setValue(cell, destination(cell) + trilerp(v000, v100, v010, v110,
											v001, v101, v011, v111,
											interpWeight[0], interpWeight[1], interpWeight[2]));
			    }
			}
		    }
		}
	    }
	});
    }

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
	for (int axis : {0,1,2})
	    assert(sourceCellLabels.getVoxelRes()[axis] % 2 == 0);

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
	for (int axis : {0,1,2})
	    assert(sourceCellLabels.getVoxelRes()[axis] % 2 == 0);

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

	UT_Vector3I voxelRes = sourceCellLabels.getVoxelRes();

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

				    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

				    if (sourceCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL ||
					sourceCellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL)
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

		    if (!localIsTileOccupiedList[tileNumber])
			localIsTileOccupiedList[tileNumber] = true;
		}

		for (int tileNumber = 0; tileNumber < tileCount; ++tileNumber)
		{
		    if (localIsTileOccupiedList[tileNumber])
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

			// Load up neighbouring INTERIOR cells that have not been visited
			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < voxelRes[axis]);

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
	    assert(coarseCellLabels.getVoxelRes()[axis] % 2 == 0 &&
		    fineCellLabels.getVoxelRes()[axis] % 2 == 0);

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
			break;

		    if (!dirichletTestPassed)
			break;

		    if (!vit.atEnd())
		    {
			if (!vit.isTileConstant() || vit.getValue() == CellLabels::DIRICHLET_CELL)
			{
			    vitt.setTile(vit);

			    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			    {
				UT_Vector3I fineCell(vitt.x(), vitt.y(), vitt.z());
				UT_Vector3I coarseCell = fineCell / 2;
				// If the fine cell is Dirichlet, it's coarse cell equivalent has to also be Dirichlet
				if (vitt.getValue() == CellLabels::DIRICHLET_CELL)
				{
				    if (coarseCellLabels(coarseCell) != CellLabels::DIRICHLET_CELL)
					dirichletTestPassed = false;
				}
				else if (vitt.getValue() == CellLabels::EXTERIOR_CELL)
				{
				    // If the fine cell is interior, the coarse cell can be either
				    // interior or Dirichlet (if a sibling cell is Dirichlet).

				    if (coarseCellLabels(coarseCell) == CellLabels::EXTERIOR_CELL)
					dirichletTestPassed = false;
				}
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
			break;

		    if (!coarseningStrategyTestPassed)
			break;

		    if (!vit.atEnd())
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
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

    StoreReal
    dotProduct(const UT_VoxelArray<StoreReal> &vectorA,
		const UT_VoxelArray<StoreReal> &vectorB,
		const UT_VoxelArray<int> &cellLabels)
    {
	assert(vectorA.getVoxelRes() == vectorB.getVoxelRes() &&
		vectorB.getVoxelRes() == cellLabels.getVoxelRes());

	const int tileCount = cellLabels.numTiles();

	UT_Array<SolveReal> tiledDotProduct;
	tiledDotProduct.setSize(tileCount);
	tiledDotProduct.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(tileCount, [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
			SolveReal localDotProduct = 0;
			
			vitt.setTile(vit);
			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
                        {
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
				localDotProduct += SolveReal(vectorA(cell)) * SolveReal(vectorB(cell));
			    }
			}

			tiledDotProduct[i] = localDotProduct;
		    }
		}
	    }
	});

	SolveReal accumulatedDotProduct = 0;
	for (int tile = 0; tile < tileCount; ++tile)
	    accumulatedDotProduct += tiledDotProduct[tile];

	return StoreReal(accumulatedDotProduct);
    }

    void
    addToVector(UT_VoxelArray<StoreReal> &destination,
		const UT_VoxelArray<StoreReal> &source,
		const UT_VoxelArray<int> &cellLabels)
    {
	assert(destination.getVoxelRes() == source.getVoxelRes() &&
		source.getVoxelRes() == cellLabels.getVoxelRes());

	UT_Interrupt *boss = UTgetInterrupt();

        UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
				destination.setValue(cell, destination(cell) + source(cell));
			    }
			}
		    }
		}
	    }
	});
    }

    StoreReal
    l2Norm(const UT_VoxelArray<StoreReal> &vector,
	    const UT_VoxelArray<int> &cellLabels)
    {
	assert(vector.getVoxelRes() == cellLabels.getVoxelRes());

	const int tileCount = cellLabels.numTiles();

	UT_Array<SolveReal> tiledSqrSum;
	tiledSqrSum.setSize(tileCount);
	tiledSqrSum.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(tileCount, [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
			SolveReal localSqr = 0;
			
			vitt.setTile(vit);
			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
                        {
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
				localSqr += SolveReal(vector(cell)) * SolveReal(vector(cell));
			    }
			}

			tiledSqrSum[i] = localSqr;
		    }
		}
	    }
	});

	SolveReal accumulatedSqrSum = 0;
	for (int tile = 0; tile < tileCount; ++tile)
	    accumulatedSqrSum += tiledSqrSum[tile];

	return StoreReal(SYSsqrt(accumulatedSqrSum));
    }

    StoreReal
    infNorm(const UT_VoxelArray<StoreReal> &vector,
	    const UT_VoxelArray<int> &cellLabels)
    {
	assert(vector.getVoxelRes() == cellLabels.getVoxelRes());

	const int tileCount = cellLabels.numTiles();

	UT_Array<SolveReal> tiledMax;
	tiledMax.setSize(tileCount);
	tiledMax.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(tileCount, [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
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
			StoreReal localMax = 0;
			
			vitt.setTile(vit);
			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
                        {
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
				localMax = SYSmax(localMax, fabs(vector(cell)));
			    }
			}

			tiledMax[i] = localMax;
		    }
		}
	    }
	});

	SolveReal gridMax = 0;
	for (int tile = 0; tile < tileCount; ++tile)
	    gridMax = SYSmax(gridMax, tiledMax[tile]);

	return gridMax;
    }
} //namespace HDK::GeometricMultiGridOperations