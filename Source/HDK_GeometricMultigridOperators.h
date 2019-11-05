#ifndef HDK_GEOMETRIC_MULTIGRID_OPERATIONS_H
#define HDK_GEOMETRIC_MULTIGRID_OPERATIONS_H

#include <SIM/SIM_FieldUtils.h>
#include <UT/UT_ParallelUtil.h>
#include <UT/UT_VoxelArray.h>

namespace HDK::GeometricMultigridOperators
{

    enum CellLabels { INTERIOR_CELL, EXTERIOR_CELL, DIRICHLET_CELL, BOUNDARY_CELL };

    //
    // Forward declaration of templated functions
    //

    // Interior smoothing operators. Handle boundaries in separate pass.

    template<typename SolveReal, typename StoreReal>
    void
    interiorJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const SolveReal dx);

    template<typename SolveReal, typename StoreReal>
    void
    interiorTiledGaussSeidelPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
					    const UT_VoxelArray<StoreReal> &rhs,
					    const UT_VoxelArray<int> &cellLabels,
					    const SolveReal dx,
					    const bool doSmoothOddTiles,
					    const bool doSmoothForward);

    // Jacobi smoothing along domain boundaries

    template<typename SolveReal, typename StoreReal>
    void
    boundaryJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const UT_Array<UT_Vector3I> &boundaryCells,
				    const SolveReal dx,
				    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights = nullptr);

    template<typename SolveReal, typename StoreReal>
    void
    applyPoissonMatrix(UT_VoxelArray<StoreReal> &destination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &cellLabels,
			const SolveReal dx,
			const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights = nullptr);

    template<typename SolveReal, typename StoreReal>
    void
    computePoissonResidual(UT_VoxelArray<StoreReal> &residual,
			    const UT_VoxelArray<StoreReal> &solution,
			    const UT_VoxelArray<StoreReal> &rhs,
			    const UT_VoxelArray<int> &cellLabels,
			    const SolveReal dx,
			    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights = nullptr);

    template<typename SolveReal, typename StoreReal>
    void
    downsample(UT_VoxelArray<StoreReal> &destination,
		const UT_VoxelArray<StoreReal> &source,
		const UT_VoxelArray<int> &destinationCellLabels,
		const UT_VoxelArray<int> &sourceCellLabels);

    template<typename SolveReal, typename StoreReal>
    void
    upsampleAndAdd(UT_VoxelArray<StoreReal> &destination,
		    const UT_VoxelArray<StoreReal> &source,
		    const UT_VoxelArray<int> &destinationCellLabels,
		    const UT_VoxelArray<int> &sourceCellLabels);

    template<typename SolveReal, typename StoreReal>
    void
    addToVector(UT_VoxelArray<StoreReal> &destination,
		const UT_VoxelArray<StoreReal> &source,
		const SolveReal scale,
		const UT_VoxelArray<int> &cellLabels);

    template<typename SolveReal, typename StoreReal>
    void
    addVectors(UT_VoxelArray<StoreReal> &destination,
		const UT_VoxelArray<StoreReal> &source,
		const UT_VoxelArray<StoreReal> &scaledSource,
		const SolveReal scale,
		const UT_VoxelArray<int> &cellLabels);

    template<typename SolveReal, typename StoreReal>
    SolveReal
    dotProduct(const UT_VoxelArray<StoreReal> &vectorA,
		const UT_VoxelArray<StoreReal> &vectorB,
		const UT_VoxelArray<int> &cellLabels);

    template<typename SolveReal, typename StoreReal>
    SolveReal
    l2Norm(const UT_VoxelArray<StoreReal> &vector,
	    const UT_VoxelArray<int> &cellLabels);
    
    template<typename SolveReal, typename StoreReal>
    SolveReal
    squaredL2Norm(const UT_VoxelArray<StoreReal> &vector,
		    const UT_VoxelArray<int> &cellLabels);

    template<typename StoreReal>
    StoreReal
    infNorm(const UT_VoxelArray<StoreReal> &vector,
	    const UT_VoxelArray<int> &cellLabels);

    template <typename IsExteriorCellFunctor, typename IsInteriorCellFunctor, typename IsDirichletCellFunctor>
    std::pair<UT_Vector3I, int>
    buildExpandedMGDomainLabels(UT_VoxelArray<int> &mgDomainCellLabels,
				const UT_VoxelArray<int> &baseDomainCellLabels,
				const IsExteriorCellFunctor &isExteriorCell,
				const IsInteriorCellFunctor &isInteriorCell,
				const IsDirichletCellFunctor &isDirichletCell);

    template<typename StoreReal>
    void
    buildExpandedBoundaryWeights(UT_VoxelArray<StoreReal> &mgBoundaryWeights,
				    const UT_VoxelArray<StoreReal> &baseBoundaryWeights,
				    const UT_VoxelArray<int> &mgDomainCellLabels,
				    const UT_Vector3I &exteriorOffset,
				    const int axis);

    template<typename StoreReal>
    void
    setBoundaryDomainLabels(UT_VoxelArray<int> &domainCellLabels,
			    const std::array<UT_VoxelArray<StoreReal>, 3> &boundaryWeights);

    UT_VoxelArray<int>
    buildCoarseCellLabels(const UT_VoxelArray<int> &sourceCellLabels);

    UT_Array<UT_Vector3I>
    buildBoundaryCells(const UT_VoxelArray<int> &sourceCellLabels,
			const int boundaryWidth);

    template<typename GridType>
    void
    uncompressTiles(UT_VoxelArray<GridType> &grid,
		    const UT_Array<bool> &isTileOccupiedList);

    template<typename GridType>
    void
    uncompressBoundaryTiles(UT_VoxelArray<GridType> &grid,
			    const UT_Array<UT_Vector3I> &boundaryCells);

    template<typename GridType>
    void
    uncompressActiveGrid(UT_VoxelArray<GridType> &grid,
			    const UT_VoxelArray<int> &domainCellLabels);

    // Test MG domains

    bool unitTestCoarsening(const UT_VoxelArray<int> &coarseCellLabels,
			    const UT_VoxelArray<int> &fineCellLabels);

    bool unitTestBoundaryCells(const UT_VoxelArray<int> &cellLabels);

    bool unitTestExteriorCells(const UT_VoxelArray<int> &cellLabels);

    // Templated MG operator implementations

    template<typename SolveReal, typename StoreReal>
    void
    interiorJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const SolveReal dx)
    {
	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	UT_VoxelArray<StoreReal> tempSolution = solution;

	// TODO: factor dx terms out
	const SolveReal gridScalar = 1. / (dx * dx);
	constexpr SolveReal dampedWeight = 2. / 3.;

	UT_Interrupt *boss = UTgetInterrupt();

        UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    UT_VoxelProbeCube<StoreReal> tempSolutionProbe;
	    tempSolutionProbe.setConstPlusArray(&tempSolution);

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> solutionProbe;
	    solutionProbe.setArray(&solution);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> rhsProbe;
	    rhsProbe.setConstArray(&rhs);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		// Ignore boundary cells so we don't have to handle irregular stencils.
		// We can clean this up during boundary smoothing
		if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL)
			{
			    tempSolutionProbe.setIndexPlus(vit);
			    solutionProbe.setIndex(vit);
			    rhsProbe.setIndex(vit);

			    SolveReal laplacian = 0;

			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I offset(0,0,0);
				    offset[axis] += direction == 0 ? -1 : 1;
				    laplacian -= tempSolutionProbe.getValue(offset[0], offset[1], offset[2]);
				}

			    laplacian += 6. * tempSolutionProbe.getValue(0, 0, 0);
			    SolveReal residual = SolveReal(rhsProbe.getValue()) - gridScalar * laplacian;
			    residual /= (6. * gridScalar);
			    solutionProbe.setValue(tempSolutionProbe.getValue(0, 0, 0) + dampedWeight * residual);
			}
		    }
		}	
	    }
	});
    }

    template<typename SolveReal, typename StoreReal>
    void
    interiorTiledGaussSeidelPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
					    const UT_VoxelArray<StoreReal> &rhs,
					    const UT_VoxelArray<int> &cellLabels,
					    const SolveReal dx,
					    const bool doSmoothOddTiles,
					    const bool doSmoothForward)
    {
	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	const SolveReal gridScalar = 1. / (dx * dx);
	const SolveReal inv_diagonal = 1. / (6. * gridScalar);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, true /* write */, true /* test for writes */> solutionProbes[3][3];

	    // Set off-center probes
	    for (int zOffset : {0,2})
		solutionProbes[1][zOffset].setArray(&solution);

	    for (int yOffset : {0,2})
		solutionProbes[yOffset][1].setArray(&solution);

	    // Set center probes
	    solutionProbes[1][1].setArray(&solution, -1, 1);

	    UT_VoxelProbe<int, true /* read */, false /* no write */, false> cellLabelProbe;
	    cellLabelProbe.setConstArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> rhsProbe;
	    rhsProbe.setConstArray(&rhs);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		{
		    // Filter out odd or even tiles
		    int tileNumber = vit.getLinearTileNum();
		    UT_Vector3i tileIndex;
		    cellLabels.linearTileToXYZ(tileNumber, tileIndex[0], tileIndex[1], tileIndex[2]);

		    int oddCount = 0;
		    for (int axis : {0,1,2})
			oddCount += tileIndex[axis];

		    bool isTileOdd = (oddCount % 2 != 0);
		    // We should only be applying to odd tiles and the count is even so skip this tile
		    if ((doSmoothOddTiles && !isTileOdd) || (!doSmoothOddTiles && isTileOdd))
			continue;

		    UT_Vector3I tileStart, tileEnd;
		    vit.getTileVoxels(tileStart,tileEnd);

		    auto gsSmoother = [&](const UT_Vector3I& cell)
		    {
			cellLabelProbe.setIndex(cell[0], cell[1], cell[2]);
			if (cellLabelProbe.getValue() == CellLabels::INTERIOR_CELL)
			{
#if !defined(NDEBUG)
			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I adjacentCell = SIM::FieldUtils::cellToCellMap(cell, axis, direction);
				    
				    // Geometric multigrid uses padding at the boundary limits so
				    // we should never end up out of bounds.
				    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < solution.getVoxelRes()[axis]);
				    assert(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
					    cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);
				}
#endif
			    // Set probe index for center first since it needs to write out the cache
			    solutionProbes[1][1].setIndex(cell[0], cell[1], cell[2]);

			    for (int yOffset : {-1,1})
				solutionProbes[yOffset + 1][1].setIndex(cell[0], cell[1] + yOffset, cell[2]);
			    for (int zOffset : {-1,1})
				solutionProbes[1][zOffset + 1].setIndex(cell[0], cell[1], cell[2] + zOffset);

			    // Compute laplacian
			    SolveReal laplacian = 0;

			    for (int xOffset : {-1,1})
				laplacian -= solutionProbes[1][1].getValue(xOffset);

			    for (int yOffset : {-1,1})
				laplacian -= solutionProbes[yOffset + 1][1].getValue();

			    for (int zOffset : {-1,1})
				laplacian -= solutionProbes[1][zOffset + 1].getValue();

			    rhsProbe.setIndex(cell[0], cell[1], cell[2]);
			    SolveReal residual = SolveReal(rhsProbe.getValue()) - gridScalar * laplacian;
			    residual *= inv_diagonal;

			    solutionProbes[1][1].setValue(residual);
			}
		    };

		    if (doSmoothForward)
		    {
			UT_Vector3I cell;
			for (cell[2] = tileStart[2]; cell[2] < tileEnd[2]; ++cell[2])
			    for (cell[1] = tileStart[1]; cell[1] < tileEnd[1]; ++cell[1])
				for (cell[0] = tileStart[0]; cell[0] < tileEnd[0]; ++cell[0])
				{
				    gsSmoother(cell);
				}
		    }
		    else
		    {
			UT_Vector3I cell;
			for (cell[2] = tileEnd[2] - 1; cell[2] >= tileStart[2]; --cell[2])
			    for (cell[1] = tileEnd[1] - 1; cell[1] >= tileStart[1]; --cell[1])
				for (cell[0] = tileEnd[0] - 1; cell[0] >= tileStart[0]; --cell[0])
				{
				    gsSmoother(cell);
				}
		    }
		}
	    }
	});
    }

    // Jacobi smoothing along domain boundaries
    
    template<typename SolveReal, typename StoreReal>
    void
    boundaryJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const UT_Array<UT_Vector3I> &boundaryCells,
				    const SolveReal dx,
				    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights)
    {
	assert(solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	if (boundaryWeights != nullptr)
	{
	    assert( (*boundaryWeights)[0].getVoxelRes()[0] - 1 == solution.getVoxelRes()[0] &&
		    (*boundaryWeights)[0].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		    (*boundaryWeights)[0].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		    (*boundaryWeights)[1].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		    (*boundaryWeights)[1].getVoxelRes()[1] - 1 == solution.getVoxelRes()[1] &&
		    (*boundaryWeights)[1].getVoxelRes()[2]     == solution.getVoxelRes()[2] &&

		    (*boundaryWeights)[2].getVoxelRes()[0]     == solution.getVoxelRes()[0] &&
		    (*boundaryWeights)[2].getVoxelRes()[1]     == solution.getVoxelRes()[1] &&
		    (*boundaryWeights)[2].getVoxelRes()[2] - 1 == solution.getVoxelRes()[2]);
	}

	const exint listSize = boundaryCells.size();

	const SolveReal gridScalar = 1. / (SolveReal(dx) * SolveReal(dx));

	constexpr SolveReal dampedWeight = 2. / 3.;

	UT_Array<StoreReal> tempSolution;
	tempSolution.setSize(listSize);

	UT_Interrupt *boss = UTgetInterrupt();

	// Apply Jacobi smoothing for boundary cell items and store in a temporary list
	UTparallelForLightItems(UT_BlockedRange<exint>(0, listSize), [&](const UT_BlockedRange<exint> &range)
	{
	    if (boss->opInterrupt())
		return;

	    UT_VoxelProbeCube<StoreReal> solutionProbe;
	    solutionProbe.setConstPlusArray(&solution);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> rhsProbe;
	    rhsProbe.setConstArray(&rhs);

	    for (exint cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
	    {
		UT_Vector3I cell = boundaryCells[cellIndex];

		solutionProbe.setIndexPlus(cell[0], cell[1], cell[2]);
		rhsProbe.setIndex(cell[0], cell[1], cell[2]);

		if (cellLabels(cell) == CellLabels::INTERIOR_CELL)
		{
		    SolveReal laplacian = 0;

		    for (int axis : {0,1,2})
			for (int direction : {0,1})
			{
#if !defined(NDEBUG)
			    UT_Vector3I adjacentCell = SIM::FieldUtils::cellToCellMap(cell, axis, direction);

			    // Geometric multigrid uses padding at the boundary limits so
			    // we should never end up out of bounds.
			    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < solution.getVoxelRes()[axis]);
			    assert(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
				    cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);
#endif

			    UT_Vector3I offset(0,0,0);
			    offset[axis] += direction == 0 ? -1 : 1;
			    laplacian -= solutionProbe.getValue(offset[0], offset[1], offset[2]);
			}

		    laplacian += 6. * solutionProbe.getValue(0,0,0);
		    SolveReal residual = SolveReal(rhsProbe.getValue()) - gridScalar * laplacian;
		    residual /= (6. * gridScalar);
		    tempSolution[cellIndex] = SolveReal(solutionProbe.getValue(0,0,0)) + dampedWeight * residual;
		}
		else
		{
		    assert(cellLabels(cell) == CellLabels::BOUNDARY_CELL);

		    SolveReal laplacian = 0;
		    SolveReal diagonal = 0;

		    for (int axis : {0,1,2})
			for (int direction : {0,1})
			{
			    UT_Vector3I adjacentCell = SIM::FieldUtils::cellToCellMap(cell, axis, direction);

			    // Geometric multigrid uses padding at the boundary limits so
			    // we should never end up out of bounds.
			    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < solution.getVoxelRes()[axis]);

			    if (cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I offset(0,0,0);
				offset[axis] += direction == 0 ? -1 : 1;
				SolveReal localValue = solutionProbe.getValue(offset[0], offset[1], offset[2]);

				if (boundaryWeights != nullptr)
				{
				    UT_Vector3I face = SIM::FieldUtils::cellToFaceMap(cell, axis, direction);
				    assert((*boundaryWeights)[axis](face) == 1);
				}

				laplacian -= localValue;
				++diagonal;
			    }
			    else if (cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL)
			    {
				UT_Vector3I offset(0,0,0);
				offset[axis] += direction == 0 ? -1 : 1;
				SolveReal localValue = solutionProbe.getValue(offset[0], offset[1], offset[2]);

				if (boundaryWeights != nullptr)
				{
				    UT_Vector3I face = SIM::FieldUtils::cellToFaceMap(cell, axis, direction);

					SolveReal weight = SolveReal((*boundaryWeights)[axis](face));
				    laplacian -= weight * localValue;
				    diagonal += weight;
				}
				else
				{
				    laplacian -= localValue;
				    ++diagonal;
				}
			    }
			    else if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
			    {
				if (boundaryWeights != nullptr)
				{
				    UT_Vector3I face = SIM::FieldUtils::cellToFaceMap(cell, axis, direction);
				    diagonal += (*boundaryWeights)[axis](face);
				}
				else ++diagonal;
			    }
			    else if (cellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL)
			    {
				if (boundaryWeights != nullptr)
				{
				    UT_Vector3I face = SIM::FieldUtils::cellToFaceMap(cell, axis, direction);
				    assert((*boundaryWeights)[axis](face) == 0);
				}
			    }
			}

		    laplacian += diagonal * solutionProbe.getValue(0,0,0);
		    SolveReal residual = SolveReal(rhsProbe.getValue()) - gridScalar * laplacian;
		    residual /= (diagonal * gridScalar);
		    tempSolution[cellIndex] = SolveReal(solutionProbe.getValue(0,0,0)) + dampedWeight * residual;
		}
	    }
	});

	// Apply updated solution in the temporary array to the solution grid
	UTparallelForLightItems(UT_BlockedRange<exint>(0, listSize), [&](const UT_BlockedRange<exint> &range)
	{
	    if (boss->opInterrupt())
		return;

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for writes */> solutionProbe;
	    solutionProbe.setArray(&solution);

	    for (exint cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
	    {
		UT_Vector3I cell = boundaryCells[cellIndex];

		solutionProbe.setIndex(cell[0], cell[1], cell[2]);

		// The tile that the cell falls into in the solution grid MUST be uncompressed
		assert(!solution.getLinearTile(solution.indexToLinearTile(cell[0], cell[1], cell[2]))->isConstant());

		solutionProbe.setValue(tempSolution[cellIndex]);
	    }
	});
    }

    template<typename SolveReal, typename StoreReal>
    void
    applyPoissonMatrix(UT_VoxelArray<StoreReal> &destination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &cellLabels,
			const SolveReal dx,
			const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights)
    {
	assert(destination.getVoxelRes() == source.getVoxelRes() &&
		source.getVoxelRes() == cellLabels.getVoxelRes());

	if (boundaryWeights != nullptr)
	{
	    assert( (*boundaryWeights)[0].getVoxelRes()[0] - 1 == source.getVoxelRes()[0] &&
		    (*boundaryWeights)[0].getVoxelRes()[1]     == source.getVoxelRes()[1] &&
		    (*boundaryWeights)[0].getVoxelRes()[2]     == source.getVoxelRes()[2] &&

		    (*boundaryWeights)[1].getVoxelRes()[0]     == source.getVoxelRes()[0] &&
		    (*boundaryWeights)[1].getVoxelRes()[1] - 1 == source.getVoxelRes()[1] &&
		    (*boundaryWeights)[1].getVoxelRes()[2]     == source.getVoxelRes()[2] &&

		    (*boundaryWeights)[2].getVoxelRes()[0]     == source.getVoxelRes()[0] &&
		    (*boundaryWeights)[2].getVoxelRes()[1]     == source.getVoxelRes()[1] &&
		    (*boundaryWeights)[2].getVoxelRes()[2] - 1 == source.getVoxelRes()[2]);
	}

	SolveReal gridScalar = 1. / (dx * dx);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    UT_VoxelProbeCube<StoreReal> sourceProbe;
	    sourceProbe.setConstPlusArray(&source);

	    UT_VoxelProbeCube<int> cellLabelProbe;
	    cellLabelProbe.setConstPlusArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for writes */> destinationProbe;
	    destinationProbe.setArray(&destination);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    for (vit.rewind(); !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL)
			{
			    sourceProbe.setIndexPlus(vit);
			    destinationProbe.setIndex(vit);

			    SolveReal laplacian = 0;
			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
#if !defined(NDEBUG)
				    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
				    UT_Vector3I adjacentCell = SIM::FieldUtils::cellToCellMap(cell, axis, direction);
				    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < destination.getVoxelRes()[axis]);
				    assert(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
					    cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);
#endif

				    UT_Vector3I offset(0,0,0);
				    offset[axis] += direction == 0 ? -1 : 1;
				    laplacian -= sourceProbe.getValue(offset[0], offset[1], offset[2]);
				}
			    
			    laplacian += SolveReal(6.) * SolveReal(sourceProbe.getValue(0,0,0));
			    laplacian *= gridScalar;
			    destinationProbe.setValue(laplacian);
			}
			else if (vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    sourceProbe.setIndexPlus(vit);
			    destinationProbe.setIndex(vit);
			    cellLabelProbe.setIndexPlus(vit);

			    SolveReal laplacian = 0;
			    SolveReal diagonal = 0;

			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I offset(0,0,0);
				    offset[axis] += direction == 0 ? -1 : 1;

#if !defined(NDEBUG)
				    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
				    UT_Vector3I adjacentCell = SIM::FieldUtils::cellToCellMap(cell, axis, direction);
				    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < destination.getVoxelRes()[axis]);
#endif

				    auto adjacentCellLabel = cellLabelProbe.getValue(offset[0], offset[1], offset[2]);

				    if (adjacentCellLabel == CellLabels::INTERIOR_CELL ||
					adjacentCellLabel == CellLabels::BOUNDARY_CELL)
				    {
					if (boundaryWeights != nullptr)
					{
					    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
					    UT_Vector3I face = SIM::FieldUtils::cellToFaceMap(cell, axis, direction);

					    laplacian -= SolveReal((*boundaryWeights)[axis](face)) * SolveReal(sourceProbe.getValue(offset[0], offset[1], offset[2]));
					    diagonal += (*boundaryWeights)[axis](face);
					}
					else
					{
					    laplacian -= sourceProbe.getValue(offset[0], offset[1], offset[2]);
					    ++diagonal;
					}
				    }
				    else if (adjacentCellLabel == CellLabels::DIRICHLET_CELL)
				    {
					if (boundaryWeights != nullptr)
					{
					    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
					    UT_Vector3I face = SIM::FieldUtils::cellToFaceMap(cell, axis, direction);
					    diagonal += (*boundaryWeights)[axis](face);
					}
					else ++diagonal;
				    }
				}

			    laplacian += diagonal * SolveReal(sourceProbe.getValue(0,0,0));
			    laplacian *= gridScalar;
			    destinationProbe.setValue(laplacian);
			}
		    }
		}
	    }
	});
    }

    template<typename SolveReal, typename StoreReal>
    void
    computePoissonResidual(UT_VoxelArray<StoreReal> &residual,
			    const UT_VoxelArray<StoreReal> &solution,
			    const UT_VoxelArray<StoreReal> &rhs,
			    const UT_VoxelArray<int> &cellLabels,
			    const SolveReal dx,
			    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights)
    {
	assert(residual.getVoxelRes() == solution.getVoxelRes() &&
		solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	residual.constant(0);

	applyPoissonMatrix<SolveReal>(residual, solution, cellLabels, dx, boundaryWeights);
	addVectors<SolveReal>(residual, rhs, residual, -1, cellLabels);
    }

    // TODO: explore a way to run a probe over restriction/prolongation operators
    template<typename SolveReal, typename StoreReal>
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

#if !defined(NDEBUG)
	for (int axis : {0,1,2})
	{
	    assert(destination.getVoxelRes()[axis] % 2 == 0);
	    assert(source.getVoxelRes()[axis] % 2 == 0);
	}
#endif

	destination.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(destinationCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&destinationCellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> sourceProbes[4][4];

	    for (int zOffset = 0; zOffset < 4; ++zOffset)
		for (int yOffset = 0; yOffset < 4; ++yOffset)
		    sourceProbes[yOffset][zOffset].setConstArray(&source, 0, 3);

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> destinationProbe;
	    destinationProbe.setArray(&destination);

	    UT_VoxelProbe<int, true /* read */, false /* no write */, false> destinationCellLabelProbe;
	    destinationCellLabelProbe.setConstArray(&destinationCellLabels);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    UT_Vector3I tileStart, tileEnd;
		    vit.getTileVoxels(tileStart,tileEnd);

		    UT_Vector3I cell;
		    for (cell[2] = tileStart[2]; cell[2] < tileEnd[2]; ++cell[2])
			for (cell[1] = tileStart[1]; cell[1] < tileEnd[1]; ++cell[1])
			    for (cell[0] = tileStart[0]; cell[0] < tileEnd[0]; ++cell[0])
			    {
				destinationCellLabelProbe.setIndex(cell[0], cell[1], cell[2]);

				CellLabels localCellLabel = CellLabels(destinationCellLabelProbe.getValue());

				if (localCellLabel == CellLabels::INTERIOR_CELL ||
				    localCellLabel == CellLabels::BOUNDARY_CELL)
				{
				    SolveReal sampleValue = 0;

				    UT_Vector3I startCell = 2 * cell - UT_Vector3I(1);

				    for (int zOffset = 0; zOffset < 4; ++zOffset)
					for (int yOffset = 0; yOffset < 4; ++yOffset)
					{
					    sourceProbes[yOffset][zOffset].setIndex(startCell[0], startCell[1] + yOffset, startCell[2] + zOffset);

					    for (int xOffset = 0; xOffset < 4; ++xOffset)
					    {
						sampleValue += restrictionWeights[xOffset] *
								restrictionWeights[yOffset] *
								restrictionWeights[zOffset] *
								SolveReal(sourceProbes[yOffset][zOffset].getValue(xOffset));

#if !defined(NDEBUG)
						if (!(sourceCellLabels(startCell + UT_Vector3I(xOffset, yOffset, zOffset)) == INTERIOR_CELL ||
							sourceCellLabels(startCell + UT_Vector3I(xOffset, yOffset, zOffset)) == BOUNDARY_CELL))
							    assert(source(startCell + UT_Vector3I(xOffset, yOffset, zOffset)) == 0);
#endif

					    }
					}

				    destinationProbe.setIndex(cell[0], cell[1], cell[2]);
				    destinationProbe.setValue(sampleValue);
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

    template<typename SolveReal, typename StoreReal>
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

#if !defined(NDEBUG)
	for (int axis : {0,1,2})
	{
	    assert(destination.getVoxelRes()[axis] % 2 == 0);
	    assert(source.getVoxelRes()[axis] % 2 == 0);
	}
#endif

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(destinationCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&destinationCellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> sourceProbes[2][2];

	    for (int zDirection : {0,1})
		for (int yDirection : {0,1})
		    sourceProbes[yDirection][zDirection].setConstArray(&source, 0, 1);

	    UT_VoxelProbe<StoreReal, true /* read */, true /* no write */, true /* test for write */> destinationProbe;
	    destinationProbe.setArray(&destination);

	    UT_VoxelProbe<int, true /* read */, false /* no write */, false> destinationCellLabelProbe;
	    destinationCellLabelProbe.setConstArray(&destinationCellLabels);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    UT_Vector3I tileStart, tileEnd;
		    vit.getTileVoxels(tileStart,tileEnd);

		    UT_Vector3I cell;
		    for (cell[2] = tileStart[2]; cell[2] < tileEnd[2]; ++cell[2])
			for (cell[1] = tileStart[1]; cell[1] < tileEnd[1]; ++cell[1])
			    for (cell[0] = tileStart[0]; cell[0] < tileEnd[0]; ++cell[0])
			    {
				destinationCellLabelProbe.setIndex(cell[0], cell[1], cell[2]);

				CellLabels localCellLabel = CellLabels(destinationCellLabelProbe.getValue());

				if (localCellLabel == CellLabels::INTERIOR_CELL ||
				    localCellLabel == CellLabels::BOUNDARY_CELL)
				{
				    UT_Vector3T<SolveReal> samplePoint = .5 * (UT_Vector3T<SolveReal>(cell) + UT_Vector3T<SolveReal>(.5)) - UT_Vector3T<SolveReal>(.5);

				    UT_Vector3I startCell = UT_Vector3I(samplePoint);

				    UT_Vector3T<SolveReal> interpWeight = samplePoint - UT_Vector3T<SolveReal>(startCell);

				    SolveReal sampleValues[2][2][2];

				    for (int zDirection : {0,1})
					for (int yDirection : {0,1})
					{
					    sourceProbes[yDirection][zDirection].setIndex(startCell[0], startCell[1] + yDirection, startCell[2] + zDirection);

					    for (int xDirection : {0,1})
					    {
						sampleValues[xDirection][yDirection][zDirection] = sourceProbes[yDirection][zDirection].getValue(xDirection);
#if !defined(NDEBUG)
						if (!(sourceCellLabels(startCell + UT_Vector3I(xDirection, yDirection, zDirection)) == INTERIOR_CELL ||
							sourceCellLabels(startCell + UT_Vector3I(xDirection, yDirection, zDirection)) == BOUNDARY_CELL))
						    assert(source(startCell + UT_Vector3I(xDirection, yDirection, zDirection)) == 0);
#endif
					    }
					}
				    
				    destinationProbe.setIndex(cell[0], cell[1], cell[2]);
				    destinationProbe.setValue(destinationProbe.getValue() + trilerp(sampleValues[0][0][0], sampleValues[1][0][0], sampleValues[0][1][0], sampleValues[1][1][0],
												    sampleValues[0][0][1], sampleValues[1][0][1], sampleValues[0][1][1], sampleValues[1][1][1],
												    interpWeight[0], interpWeight[1], interpWeight[2]));
				}
			    }
		}
	    }
	});
    }

    template<typename SolveReal, typename StoreReal>
    SolveReal
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

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> vectorAProbe;
	    vectorAProbe.setConstArray(&vectorA);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> vectorBProbe;
	    vectorBProbe.setConstArray(&vectorB);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    SolveReal localDotProduct = 0;

		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL ||
			    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    vectorAProbe.setIndex(vit);
			    vectorBProbe.setIndex(vit);

			    localDotProduct += SolveReal(vectorAProbe.getValue()) * SolveReal(vectorBProbe.getValue());
			}
		    }

		    tiledDotProduct[i] = localDotProduct;
		}
	    }
	});

	SolveReal accumulatedDotProduct = 0;
	for (int tile = 0; tile < tileCount; ++tile)
	    accumulatedDotProduct += tiledDotProduct[tile];

	return accumulatedDotProduct;
    }
  
    template<typename SolveReal, typename StoreReal>
    void
    addToVector(UT_VoxelArray<StoreReal> &destination,
		const UT_VoxelArray<StoreReal> &source,
		const SolveReal scale,
		const UT_VoxelArray<int> &cellLabels)
    {
	assert(destination.getVoxelRes() == source.getVoxelRes() &&
		source.getVoxelRes() == cellLabels.getVoxelRes());

	UT_Interrupt *boss = UTgetInterrupt();

        UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, true /* write */, true /* test for write */> destinationProbe;
	    destinationProbe.setArray(&destination);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> sourceProbe;
	    sourceProbe.setConstArray(&source);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    for (vit.rewind(); !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL ||
			    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    destinationProbe.setIndex(vit);
			    sourceProbe.setIndex(vit);

			    destinationProbe.setValue(SolveReal(destinationProbe.getValue()) + scale * SolveReal(sourceProbe.getValue()));
			}
		    }
		}
	    }
	});
    }

    template<typename SolveReal, typename StoreReal>
    void
    addVectors(UT_VoxelArray<StoreReal> &destination,
		const UT_VoxelArray<StoreReal> &source,
		const UT_VoxelArray<StoreReal> &scaledSource,
		const SolveReal scale,
		const UT_VoxelArray<int> &cellLabels)
    {
	assert(destination.getVoxelRes() == source.getVoxelRes() &&
		source.getVoxelRes() == scaledSource.getVoxelRes() &&
		scaledSource.getVoxelRes() == cellLabels.getVoxelRes());

	UT_Interrupt *boss = UTgetInterrupt();

        UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> destinationProbe;
	    destinationProbe.setArray(&destination);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> sourceProbe;
	    sourceProbe.setConstArray(&source);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false> scaledSourceProbe;
	    scaledSourceProbe.setConstArray(&scaledSource);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    for (vit.rewind(); !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL ||
			    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    destinationProbe.setIndex(vit);
			    sourceProbe.setIndex(vit);
			    scaledSourceProbe.setIndex(vit);

			    destinationProbe.setValue(SolveReal(sourceProbe.getValue()) + scale * SolveReal(scaledSourceProbe.getValue()));
			}
		    }
		}
	    }
	});
    }

    template<typename SolveReal, typename StoreReal>
    SolveReal
    l2Norm(const UT_VoxelArray<StoreReal> &vector,
	    const UT_VoxelArray<int> &cellLabels)
    {
	return SYSsqrt(squaredL2Norm<SolveReal>(vector, cellLabels));
    }

    template<typename SolveReal, typename StoreReal>
    SolveReal
    squaredL2Norm(const UT_VoxelArray<StoreReal> &vector,
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

	    UT_VoxelProbe<StoreReal, true /* read */, false /* write */, false> vectorProbe;
	    vectorProbe.setConstArray(&vector);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    SolveReal localSqr = 0;

		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL ||
			    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    vectorProbe.setIndex(vit);

			    SolveReal value = vectorProbe.getValue();
			    localSqr += (value * value);
			}
		    }

		    tiledSqrSum[i] = localSqr;
		}
	    }
	});

	SolveReal accumulatedSqrSum = 0;
	for (int tile = 0; tile < tileCount; ++tile)
	    accumulatedSqrSum += tiledSqrSum[tile];

	return accumulatedSqrSum;
    }

    template<typename StoreReal>
    StoreReal
    infNorm(const UT_VoxelArray<StoreReal> &vector,
	    const UT_VoxelArray<int> &cellLabels)
    {
	assert(vector.getVoxelRes() == cellLabels.getVoxelRes());

	const int tileCount = cellLabels.numTiles();

	UT_Array<StoreReal> tiledMax;
	tiledMax.setSize(tileCount);
	tiledMax.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(tileCount, [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* write */, false> vectorProbe;
	    vectorProbe.setConstArray(&vector);

	    for (int i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    StoreReal localMax = 0;

		    for (vit.rewind(); !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL ||
			    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    vectorProbe.setIndex(vit);

			    localMax = SYSmax(localMax, vectorProbe.getValue());
			}
		    }

		    tiledMax[i] = localMax;
		}
	    }
	});

	StoreReal gridMax = 0;
	for (int tile = 0; tile < tileCount; ++tile)
	    gridMax = SYSmax(gridMax, tiledMax[tile]);

	return gridMax;
    }    

    template <typename IsExteriorCellFunctor, typename IsInteriorCellFunctor, typename IsDirichletCellFunctor>
    std::pair<UT_Vector3I, int>
    buildExpandedMGDomainLabels(UT_VoxelArray<int> &mgDomainCellLabels,
				const UT_VoxelArray<int> &baseDomainCellLabels,
				const IsExteriorCellFunctor &isExteriorCell,
				const IsInteriorCellFunctor &isInteriorCell,
				const IsDirichletCellFunctor &isDirichletCell)
    {
	// Build domain labels with the appropriate padding to apply
	// geometric multigrid directly without a wasteful transfer
	// for each v-cycle.

	// Cap MG levels at 4 voxels in the smallest dimension
	fpreal minLog = std::min(std::log2(fpreal(baseDomainCellLabels.getVoxelRes()[0])),
				    std::log2(fpreal(baseDomainCellLabels.getVoxelRes()[1])));
	minLog = std::min(minLog, std::log2(fpreal(baseDomainCellLabels.getVoxelRes()[2])));

	int mgLevels = ceil(minLog) - std::log2(fpreal(2));

	// Add the necessary exterior cells so that after coarsening to the top level
	// there is still a single layer of exterior cells
	int exteriorPadding = std::pow(2, mgLevels - 1);

	UT_Vector3I expandedResolution = baseDomainCellLabels.getVoxelRes() + 2 * UT_Vector3I(exteriorPadding);

	// Expand the domain to be a power of 2.
	for (int axis : {0,1,2})
	{
	    fpreal logSize = std::log2(fpreal(expandedResolution[axis]));
	    logSize = std::ceil(logSize);

	    expandedResolution[axis] = exint(std::exp2(logSize));
	}
    
	UT_Vector3I exteriorOffset = UT_Vector3I(exteriorPadding);

	mgDomainCellLabels.size(expandedResolution[0], expandedResolution[1], expandedResolution[2]);
	mgDomainCellLabels.constant(CellLabels::EXTERIOR_CELL);

	// Build domain cell labels
	UT_Interrupt *boss = UTgetInterrupt();

	// Uncompress internal domain label tiles
	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(mgDomainCellLabels.numTiles());
	isTileOccupiedList.constant(false);

	UTparallelForEachNumber(baseDomainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&baseDomainCellLabels);
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
		    if (!vit.isTileConstant() || !isExteriorCell(vit.getValue()))
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (!isExteriorCell(vitt.getValue()))
			    {
				UT_Vector3I cell = UT_Vector3I(vitt.x(), vitt.y(), vitt.z()) + exteriorOffset;

				int tileNumber = mgDomainCellLabels.indexToLinearTile(cell[0], cell[1], cell[2]);
				if (!isTileOccupiedList[tileNumber])
				    isTileOccupiedList[tileNumber] = true;
			    }
			}
		    }
		}
	    }
	});

	uncompressTiles(mgDomainCellLabels, isTileOccupiedList);

	// Copy initial domain labels to interior domain labels with padding
	UTparallelForEachNumber(baseDomainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&baseDomainCellLabels);

	    UT_VoxelTileIterator<int> vitt;

	    UT_VoxelProbe<int, false /* no read */, true /* write */, true /* test for write */> localDomainProbe;
	    localDomainProbe.setArray(&mgDomainCellLabels);

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.atEnd())
		{
		    if (!vit.isTileConstant() || !isExteriorCell(vit.getValue()))
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    const exint domainValue = vitt.getValue();
			    if (!isExteriorCell(domainValue))
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
				UT_Vector3I expandedCell = cell + exteriorOffset;

				assert(!mgDomainCellLabels.getLinearTile(mgDomainCellLabels.indexToLinearTile(expandedCell[0],
														expandedCell[1],
														expandedCell[2]))->isConstant());

				localDomainProbe.setIndex(expandedCell[0], expandedCell[1], expandedCell[2]);

				if (isInteriorCell(domainValue))
				    localDomainProbe.setValue(CellLabels::INTERIOR_CELL);
				else
				{
				    assert(isDirichletCell(domainValue) && !isExteriorCell(domainValue));
				    localDomainProbe.setValue(CellLabels::DIRICHLET_CELL);
				}
			    }
			}
		    }
		}
	    }
	});

	mgDomainCellLabels.collapseAllTiles();

	return std::pair<UT_Vector3I, int>(exteriorOffset, mgLevels);
    }

    template<typename StoreReal>
    void
    buildExpandedBoundaryWeights(UT_VoxelArray<StoreReal> &mgBoundaryWeights,
				    const UT_VoxelArray<StoreReal> &baseBoundaryWeights,
				    const UT_VoxelArray<int> &mgDomainCellLabels,
				    const UT_Vector3I &exteriorOffset,
				    const int axis)
    {
	using SIM::FieldUtils::setFieldValue;
	using SIM::FieldUtils::getFieldValue;
	using SIM::FieldUtils::faceToCellMap;

#if !defined(NDEBUG)	
	UT_Vector3I boundaryVoxelRes = mgBoundaryWeights.getVoxelRes();
	--boundaryVoxelRes[axis];
	assert(boundaryVoxelRes == mgDomainCellLabels.getVoxelRes());

	assert(mgBoundaryWeights.getVoxelRes()[0] >= baseBoundaryWeights.getVoxelRes()[0]  + exteriorOffset[0] &&
		mgBoundaryWeights.getVoxelRes()[1] >= baseBoundaryWeights.getVoxelRes()[1]  + exteriorOffset[1] &&
		mgBoundaryWeights.getVoxelRes()[2] >= baseBoundaryWeights.getVoxelRes()[2] + exteriorOffset[2]);
#endif
	// Make sure weights are empty
	mgBoundaryWeights.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(mgBoundaryWeights.numTiles());
	isTileOccupiedList.constant(false);

	UTparallelForEachNumber(baseBoundaryWeights.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<StoreReal> vit;
	    vit.setConstArray(&baseBoundaryWeights);

	    UT_VoxelTileIterator<StoreReal> vitt;

	    if (boss->opInterrupt())
		return;
	 
	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (!vit.atEnd())
		{
		    if (!vit.isTileConstant() || vit.getValue() > 0)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{	      
			    if (vitt.getValue() > 0)
			    {
				UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());

				UT_Vector3I expandedFace = face + exteriorOffset;

				int tileNumber = mgBoundaryWeights.indexToLinearTile(expandedFace[0], expandedFace[1], expandedFace[2]);

				if (!isTileOccupiedList[tileNumber])
				    isTileOccupiedList[tileNumber] = true;
			    }
			}
		    }
		}
	    }
	});
    
	uncompressTiles(mgBoundaryWeights, isTileOccupiedList);

	UTparallelForEachNumber(baseBoundaryWeights.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<StoreReal> vit;
	    vit.setConstArray(&baseBoundaryWeights);

	    UT_VoxelTileIterator<StoreReal> vitt;

	    if (boss->opInterrupt())
		return;

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for writes */> mgWeightsProbe;
	    mgWeightsProbe.setArray(&mgBoundaryWeights);
	 
	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (!vit.atEnd())
		{
		    if (!vit.isTileConstant() || vit.getValue() > 0)
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{	      
			    if (vitt.getValue() > 0)
			    {
				UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());

#if !defined(NDEBUG)
				UT_Vector3I backwardCell = faceToCellMap(face, axis, 0);
				UT_Vector3I forwardCell = faceToCellMap(face, axis, 1);

				assert(mgDomainCellLabels(backwardCell + exteriorOffset) != CellLabels::EXTERIOR_CELL &&
					mgDomainCellLabels(forwardCell + exteriorOffset) != CellLabels::EXTERIOR_CELL);

				assert(mgDomainCellLabels(backwardCell + exteriorOffset) == CellLabels::INTERIOR_CELL ||
					mgDomainCellLabels(forwardCell + exteriorOffset) == CellLabels::INTERIOR_CELL);
#endif

				UT_Vector3I expandedFace = face + exteriorOffset;
				assert(!mgBoundaryWeights.getLinearTile(mgBoundaryWeights.indexToLinearTile(expandedFace[0],
													    expandedFace[1],
													    expandedFace[2]))->isConstant());

				mgWeightsProbe.setIndex(face[0] + exteriorOffset[0],
							face[1] + exteriorOffset[1],
							face[2] + exteriorOffset[2]);

				mgWeightsProbe.setValue(vitt.getValue());
			    }
			}
		    }
		}
	    }
	});
    }

    template<typename StoreReal>
    void
    setBoundaryDomainLabels(UT_VoxelArray<int> &domainCellLabels,
			    const std::array<UT_VoxelArray<StoreReal>, 3> &boundaryWeights)
    {
	using SIM::FieldUtils::cellToCellMap;
	using SIM::FieldUtils::cellToFaceMap;
	using HDK::GeometricMultigridOperators::CellLabels;

	UT_Interrupt *boss = UTgetInterrupt();

	// Now set INTERIOR cells with DIRICHLET or EXTERIOR neighbours OR a non-unity boundary weight to BOUNDARY
	UT_Array<UT_Array<UT_Vector3I>> boundaryCellList;
	boundaryCellList.setSize(domainCellLabels.numTiles());

	UTparallelForEachNumber(domainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&domainCellLabels);

	    UT_VoxelTileIterator<int> vitt;

	    for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	    {
		vit.myTileStart = tileNumber;
		vit.myTileEnd = tileNumber + 1;
		vit.rewind();

		UT_Array<UT_Vector3I> &localBoundaryCellList = boundaryCellList[tileNumber];

		if (boss->opInterrupt())
		    break;

		if (!vit.atEnd())
		{
		    if (!vit.isTileConstant() ||
			vit.getValue() == CellLabels::INTERIOR_CELL)
		    {
			vitt.setTile(vit);
			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL)
			    {
				UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				bool isBoundaryCell = false;

				for (int axis = 0; axis < 3 & !isBoundaryCells; ++axis)
				    for (int direction : {0,1})
				    {
					UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

					assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < domainCellLabels.getVoxelRes()[axis]);

					if (domainCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL ||
					    domainCellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL)
					{
					    isBoundaryCell = true;
					    break;
					}

					// Check boundary weight
					UT_Vector3I face = cellToFaceMap(cell, axis, direction);

					if (boundaryWeights[axis](face) != 1)
					{
					    isBoundaryCell = true;
					    break;
					}
				    }

				if (hasBoundaryNeighbour)
				    localBoundaryCellList.append(cell);
			    }
			}
		    }
		}
	    }
	});

	UTparallelForEachNumber(domainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	    {
		UT_Array<UT_Vector3I> &localBoundaryCellList = boundaryCellList[tileNumber];

		for (int cellIndex = 0; cellIndex < localBoundaryCellList.size(); ++cellIndex)
		{
		    UT_Vector3I cell = localBoundaryCellList[cellIndex];
		    assert(domainCellLabels(cell) == CellLabels::INTERIOR_CELL);

		    domainCellLabels.setValue(cell, CellLabels::BOUNDARY_CELL);
		}
	    }
	});
    }


    template<typename GridType>
    void
    uncompressTiles(UT_VoxelArray<GridType> &grid,
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

    template<typename GridType>
    void
    uncompressBoundaryTiles(UT_VoxelArray<GridType> &grid,
			    const UT_Array<UT_Vector3I> &boundaryCells)
    {
	// We need to uncompress tiles in solution grids that boundary cells sit on.
	// This isn't immediately obvious because there will be a smoother pass over
	// the entire solution domain before applying boundary smoothing.
	// However, if the residual happens to be zero along the boundary, the tiles will
	// not be expanded safely during the domain smoothing and could cause a seg fault
	// when expanding during boundary smoothing.

	UT_Interrupt *boss = UTgetInterrupt();

	const int tileCount = grid.numTiles();
	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(tileCount);
	isTileOccupiedList.constant(false);

	UTparallelForLightItems(UT_BlockedRange<exint>(0, boundaryCells.size()), [&](const UT_BlockedRange<exint> &range)
	{
	    if (boss->opInterrupt())
		return;

	    UT_Array<bool> localIsTileOccupiedList;
	    localIsTileOccupiedList.setSize(tileCount);
	    localIsTileOccupiedList.constant(false);

	    for (exint cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
	    {
		UT_Vector3I cell = boundaryCells[cellIndex];

		int tileNumber = grid.indexToLinearTile(cell[0], cell[1], cell[2]);

		if (!localIsTileOccupiedList[tileNumber])
		    localIsTileOccupiedList[tileNumber] = true;
	    }

	    for (int tileNumber = 0; tileNumber < tileCount; ++tileNumber)
	    {
		if (localIsTileOccupiedList[tileNumber] && !isTileOccupiedList[tileNumber])
		    isTileOccupiedList[tileNumber] = true;   
	    }
	});

	uncompressTiles(grid, isTileOccupiedList);
    }
    
    template<typename GridType>
    void
    uncompressActiveGrid(UT_VoxelArray<GridType> &grid,
			    const UT_VoxelArray<int> &domainCellLabels)
    {
	assert(grid.getVoxelRes() == domainCellLabels.getVoxelRes());

	UT_Interrupt *boss = UTgetInterrupt();

	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(domainCellLabels.numTiles());
	isTileOccupiedList.constant(false);

	UTparallelForEachNumber(domainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&domainCellLabels);
	    UT_VoxelTileIterator<int> vitt;

	    for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	    {
		vit.myTileStart = tileNumber;
		vit.myTileEnd = tileNumber + 1;
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
			    if (vitt.getValue() == CellLabels::INTERIOR_CELL ||
				vitt.getValue() == CellLabels::BOUNDARY_CELL)
			    {
				UT_Vector3I cell = UT_Vector3I(vitt.x(), vitt.y(), vitt.z());

				assert(tileNumber == domainCellLabels.indexToLinearTile(cell[0], cell[1], cell[2]) &&
					tileNumber == grid.indexToLinearTile(cell[0], cell[1], cell[2]));

				if (!isTileOccupiedList[tileNumber])
				    isTileOccupiedList[tileNumber] = true;
			    }
			}
		    }
		    else if (vit.getValue() == CellLabels::INTERIOR_CELL ||
				vit.getValue() == CellLabels::BOUNDARY_CELL)
		    {
			if (!isTileOccupiedList[tileNumber])
			    isTileOccupiedList[tileNumber] = true;
		    }
		}
	    }
	});

	uncompressTiles(grid, isTileOccupiedList);
    }

} // HDK::GeometricMultigridOperators

#endif