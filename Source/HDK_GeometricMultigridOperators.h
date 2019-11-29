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

    // Interior smoothing operators.

    template<typename SolveReal, typename SourceFunctor, typename CellLabelFunctor, typename BoundaryWeightsFunctor>
    std::pair<SolveReal, SolveReal>
    computeLaplacian(const SourceFunctor &sourceFunctor,
			const CellLabelFunctor &cellLabelsFunctor,
			const BoundaryWeightsFunctor &boundaryWeightsFunctor,
			bool applyBoundaryWeights);

    template<typename SolveReal, typename StoreReal>
    void
    jacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
			    const UT_VoxelArray<StoreReal> &rhs,
			    const UT_VoxelArray<int> &cellLabels,
			    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights = nullptr);

    template<typename SolveReal, typename StoreReal>
    void
    tiledGaussSeidelPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const bool doSmoothOddTiles,
				    const bool doSmoothForward,
				    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights = nullptr);

    // Jacobi smoothing along domain boundaries

    template<typename SolveReal, typename StoreReal>
    void
    boundaryJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const UT_Array<UT_Vector3I> &boundaryCells,
				    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights = nullptr);

    template<typename SolveReal, typename StoreReal>
    void
    applyPoissonMatrix(UT_VoxelArray<StoreReal> &destination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &cellLabels,
			const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights = nullptr);

    template<typename SolveReal, typename StoreReal>
    void
    computePoissonResidual(UT_VoxelArray<StoreReal> &residual,
			    const UT_VoxelArray<StoreReal> &solution,
			    const UT_VoxelArray<StoreReal> &rhs,
			    const UT_VoxelArray<int> &cellLabels,
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

    template<typename StoreReal>
    void
    scaleVector(UT_VoxelArray<StoreReal> &vector,
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
    buildExpandedCellLabels(UT_VoxelArray<int> &expandedCellLabels,
			    const UT_VoxelArray<int> &baseCellLabels,
			    const IsExteriorCellFunctor &isExteriorCell,
			    const IsInteriorCellFunctor &isInteriorCell,
			    const IsDirichletCellFunctor &isDirichletCell);

    template<typename StoreReal>
    void
    buildExpandedBoundaryWeights(UT_VoxelArray<StoreReal> &expandedBoundaryWeights,
				    const UT_VoxelArray<StoreReal> &baseBoundaryWeights,
				    const UT_VoxelArray<int> &expandedCellLabels,
				    const UT_Vector3I &exteriorOffset,
				    const int axis);

    template<typename StoreReal>
    void
    setBoundaryCellLabels(UT_VoxelArray<int> &cellLabels,
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

    template<typename StoreReal>
    bool unitTestBoundaryCells(const UT_VoxelArray<int> &cellLabels,
				const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights = nullptr);

    bool unitTestExteriorCells(const UT_VoxelArray<int> &cellLabels);

    // Templated MG operator implementations
    template<typename SolveReal, typename SourceFunctor, typename CellLabelFunctor, typename BoundaryWeightsFunctor>
    std::pair<SolveReal, SolveReal>
    computeLaplacian(const SourceFunctor &sourceFunctor,
			const CellLabelFunctor &cellLabelsFunctor,
			const BoundaryWeightsFunctor &boundaryWeightsFunctor,
			bool applyBoundaryWeights)
    {
	using SIM::FieldUtils::cellToCellMap;

	SolveReal laplacian = 0;
	SolveReal diagonal = 0;

	UT_Vector3I centerSample(0,0,0);

	if (cellLabelsFunctor(centerSample) == CellLabels::INTERIOR_CELL)
	{
	    for (int axis : {0,1,2})
		for (int direction : {0,1})
		{
		    UT_Vector3I offset = cellToCellMap(centerSample, axis, direction);
		    assert(cellLabelsFunctor(offset) == CellLabels::INTERIOR_CELL ||
			    cellLabelsFunctor(offset) == CellLabels::BOUNDARY_CELL);

		    if (applyBoundaryWeights)
			assert(boundaryWeightsFunctor(axis, direction) == 1);

		    laplacian -= SolveReal(sourceFunctor(offset));
		}

	    diagonal = 6;
	}
	else
	{
	    assert(cellLabelsFunctor(centerSample) == CellLabels::BOUNDARY_CELL);

	    for (int axis : {0,1,2})
		for (int direction : {0,1})
		{
		    UT_Vector3I offset = cellToCellMap(centerSample, axis, direction);

		    auto adjacentCellLabel = cellLabelsFunctor(offset);

		    if (adjacentCellLabel == CellLabels::INTERIOR_CELL)
		    {
			if (applyBoundaryWeights)
			    assert(boundaryWeightsFunctor(axis, direction) == 1);

			laplacian -= SolveReal(sourceFunctor(offset));
			++diagonal;
		    }
		    else if (adjacentCellLabel == CellLabels::BOUNDARY_CELL)
		    {
			SolveReal sourceValue = sourceFunctor(offset);

			if (applyBoundaryWeights)
			{
			    SolveReal weight = boundaryWeightsFunctor(axis, direction);
			    laplacian -= weight * sourceValue;
			    diagonal += weight;
			}
			else
			{
			    laplacian -= sourceValue;
			    ++diagonal;
			}
		    }
		    else if (adjacentCellLabel == CellLabels::DIRICHLET_CELL)
		    {
			if (applyBoundaryWeights)
			    diagonal += boundaryWeightsFunctor(axis, direction);
			else
			    ++diagonal;
		    }
		    else
		    {
			if (applyBoundaryWeights)
			    assert(boundaryWeightsFunctor(axis, direction) == 0);
		    }
		}
	}

	laplacian += diagonal * SolveReal(sourceFunctor(centerSample));
	return std::pair<SolveReal, SolveReal>(laplacian, diagonal);
    }

    template<typename SolveReal, typename StoreReal>
    void
    jacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
			    const UT_VoxelArray<StoreReal> &rhs,
			    const UT_VoxelArray<int> &cellLabels,
			    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights)
    {
	using SIM::FieldUtils::cellToFaceMap;

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

	UT_VoxelArray<StoreReal> tempSolution = solution;

	constexpr SolveReal dampedWeight = 2. / 3.;

	UT_Interrupt *boss = UTgetInterrupt();

        UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    // Build probes
	    UT_VoxelProbeCube<StoreReal> tempSolutionProbe;
	    tempSolutionProbe.setConstPlusArray(&tempSolution);

	    UT_VoxelProbeCube<int> cellLabelsProbe;
	    cellLabelsProbe.setConstPlusArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, true /* write */, true /* test for writes */> solutionProbe;
	    solutionProbe.setArray(&solution);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false /* no test for writes */> rhsProbe;
	    rhsProbe.setConstArray(&rhs);

	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

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
		    for (; !vit.atEnd(); vit.advance())
		    {
			auto cellLabel = vit.getValue();
			if (cellLabel == CellLabels::INTERIOR_CELL ||
			    cellLabel == CellLabels::BOUNDARY_CELL)
			{
			    UT_Vector3I cell(vit.x(), vit.y(), vit.z());

			    tempSolutionProbe.setIndexPlus(cell[0], cell[1], cell[2]);
			    cellLabelsProbe.setIndexPlus(cell[0], cell[1], cell[2]);

			    auto sourceFunctor = [&](const UT_Vector3I &offset) { return tempSolutionProbe.getValue(offset); };
			    auto cellLabelFunctor = [&](const UT_Vector3I &offset) { return cellLabelsProbe.getValue(offset); };

			    auto boundaryWeightsFunctor = [&](const int axis, const int direction)
			    {
				UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				return (*boundaryWeights)[axis](face);
			    };

			    std::pair<SolveReal, SolveReal> laplacianResults = computeLaplacian<SolveReal>(sourceFunctor, cellLabelFunctor, boundaryWeightsFunctor, boundaryWeights != nullptr);

			    SolveReal laplacian = laplacianResults.first;
			    SolveReal diagonal = laplacianResults.second;

			    if (cellLabel == CellLabels::INTERIOR_CELL)
				assert(diagonal == 6.);
			    else
				assert(diagonal > 0);

			    rhsProbe.setIndex(cell[0], cell[1], cell[2]);
			    SolveReal residual = SolveReal(rhsProbe.getValue()) - laplacian;
			    residual /= diagonal;

			    solutionProbe.setIndex(cell[0], cell[1], cell[2]);
			    solutionProbe.setValue(solutionProbe.getValue() + dampedWeight * residual);
			}
		    }
		}	
	    }
	});
    }

    template<typename SolveReal, typename StoreReal>
    void
    tiledGaussSeidelPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
				    const UT_VoxelArray<StoreReal> &rhs,
				    const UT_VoxelArray<int> &cellLabels,
				    const bool doSmoothOddTiles,
				    const bool doSmoothForward,
				    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights)
    {
	using SIM::FieldUtils::cellToFaceMap;

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

	    UT_VoxelProbeCube<int> cellLabelsProbe;
	    cellLabelsProbe.setConstPlusArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, false /* no write */, false /* no test for writes */> rhsProbe;

	    rhsProbe.setConstArray(&rhs);
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
		    // Filter out odd or even tiles
		    int tileNumber = vit.getLinearTileNum();
		    UT_Vector3i tileIndex;
		    cellLabels.linearTileToXYZ(tileNumber, tileIndex[0], tileIndex[1], tileIndex[2]);

		    int oddCount = 0;
		    for (int axis : {0, 1, 2})
			oddCount += tileIndex[axis];

		    bool isTileOdd = (oddCount % 2 != 0);
		    // We should only be applying to odd tiles and the count is even so skip this tile
		    if ((doSmoothOddTiles && !isTileOdd) || (!doSmoothOddTiles && isTileOdd))
			continue;

		    UT_Vector3I tileStart, tileEnd;
		    vit.getTileVoxels(tileStart,tileEnd);

		    auto gsSmoother = [&](const UT_Vector3I& cell)
		    {
			cellLabelsProbe.setIndexPlus(cell[0], cell[1], cell[2]);

			auto cellLabel = cellLabelsProbe.getValue(UT_Vector3I(0,0,0));

			if (cellLabel == CellLabels::INTERIOR_CELL ||
			    cellLabel == CellLabels::BOUNDARY_CELL)
			{
			    // Set probe index for center first since it needs to write out the cache
			    solutionProbes[1][1].setIndex(cell[0], cell[1], cell[2]);

			    for (int yOffset : {-1,1})
				solutionProbes[yOffset + 1][1].setIndex(cell[0], cell[1] + yOffset, cell[2]);
			    for (int zOffset : {-1,1})
				solutionProbes[1][zOffset + 1].setIndex(cell[0], cell[1], cell[2] + zOffset);

			    auto sourceFunctor = [&](const UT_Vector3I &offset) { return solutionProbes[offset[1] + 1][offset[2] + 1].getValue(offset[0]); };
			    auto cellLabelFunctor = [&](const UT_Vector3I &offset) { return cellLabelsProbe.getValue(offset); };

			    auto boundaryWeightsFunctor = [&](const int axis, const int direction)
			    {
				UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				return (*boundaryWeights)[axis](face);
			    };

			    std::pair<SolveReal, SolveReal> laplacianResults = computeLaplacian<SolveReal>(sourceFunctor, cellLabelFunctor, boundaryWeightsFunctor, boundaryWeights != nullptr);

			    SolveReal laplacian = laplacianResults.first;
			    SolveReal diagonal = laplacianResults.second;

			    if (cellLabel == CellLabels::INTERIOR_CELL)
				assert(diagonal == 6.);
			    else
				assert(diagonal > 0);

			    rhsProbe.setIndex(cell[0], cell[1], cell[2]);
			    SolveReal residual = SolveReal(rhsProbe.getValue()) - laplacian;
			    residual /= diagonal;

			    solutionProbes[1][1].setValue(solutionProbes[1][1].getValue(0) + residual);
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
				    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights)
    {
	using SIM::FieldUtils::cellToFaceMap;

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

	constexpr SolveReal dampedWeight = 2. / 3.;

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

		auto sourceFunctor = [&](const UT_Vector3I &offset) { return solution(cell + offset); };
		auto cellLabelFunctor = [&](const UT_Vector3I &offset) { return cellLabels(cell + offset); };

		auto boundaryWeightsFunctor = [&](const int axis, const int direction)
		{
		    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
		    return (*boundaryWeights)[axis](face);
		};

		std::pair<SolveReal, SolveReal> laplacianResults = computeLaplacian<SolveReal>(sourceFunctor, cellLabelFunctor, boundaryWeightsFunctor, boundaryWeights != nullptr);

		SolveReal laplacian = laplacianResults.first;
		SolveReal diagonal = laplacianResults.second;

#if !defined(NDEBUG)
		auto cellLabel = cellLabels(cell);
		if (cellLabel == CellLabels::INTERIOR_CELL)
		    assert(diagonal == 6.);
		else
		{
		    assert(cellLabel == CellLabels::BOUNDARY_CELL);
		    assert(diagonal > 0);
		}
#endif

		SolveReal residual = SolveReal(rhs(cell)) - laplacian;
		residual /= diagonal;

		tempSolution[cellIndex] = SolveReal(solution(cell)) + dampedWeight * residual;
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

    template<typename SolveReal, typename StoreReal>
    void
    applyPoissonMatrix(UT_VoxelArray<StoreReal> &destination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &cellLabels,
			const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights)
    {
	using SIM::FieldUtils::cellToFaceMap;

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

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    // Build probes
	    UT_VoxelProbeCube<StoreReal> sourceProbe;
	    sourceProbe.setConstPlusArray(&source);

	    UT_VoxelProbeCube<int> cellLabelsProbe;
	    cellLabelsProbe.setConstPlusArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for writes */ > destinationProbe;
	    destinationProbe.setArray(&destination);

	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);
	    
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
		    for (; !vit.atEnd(); vit.advance())
		    {
			auto cellLabel = vit.getValue();
			if (cellLabel == CellLabels::INTERIOR_CELL ||
			    cellLabel == CellLabels::BOUNDARY_CELL)
			{
			    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
			    
			    sourceProbe.setIndexPlus(cell[0], cell[1], cell[2]);
			    cellLabelsProbe.setIndexPlus(cell[0], cell[1], cell[2]);

			    auto sourceFunctor = [&](const UT_Vector3I &offset) { return sourceProbe.getValue(offset); };
			    auto cellLabelFunctor = [&](const UT_Vector3I &offset) { return cellLabelsProbe.getValue(offset); };

			    auto boundaryWeightsFunctor = [&](const int axis, const int direction)
			    {
				UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				return (*boundaryWeights)[axis](face);
			    };

			    std::pair<SolveReal, SolveReal> laplacianResults = computeLaplacian<SolveReal>(sourceFunctor, cellLabelFunctor, boundaryWeightsFunctor, boundaryWeights != nullptr);

			    SolveReal laplacian = laplacianResults.first;
			    SolveReal diagonal = laplacianResults.second;

			    if (cellLabel == CellLabels::INTERIOR_CELL)
				assert(diagonal == 6.);
			    else
				assert(diagonal > 0);

			    destinationProbe.setIndex(cell[0], cell[1], cell[2]);
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
			    const std::array<UT_VoxelArray<StoreReal>, 3>  *boundaryWeights)
    {
	assert(residual.getVoxelRes() == solution.getVoxelRes() &&
		solution.getVoxelRes() == rhs.getVoxelRes() &&
		rhs.getVoxelRes() == cellLabels.getVoxelRes());

	residual.constant(0);

	applyPoissonMatrix<SolveReal>(residual, solution, cellLabels, boundaryWeights);
	addVectors<SolveReal>(residual, rhs, residual, -1, cellLabels);
    }

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

		    for (; !vit.atEnd(); vit.advance())
		    {
			auto destinationLabel = vit.getValue();
			
			if (destinationLabel == CellLabels::INTERIOR_CELL ||
			    destinationLabel == CellLabels::BOUNDARY_CELL)
			{
			    SolveReal sampleValue = 0;

			    UT_Vector3I cell(vit.x(), vit.y(), vit.z());

			    UT_Vector3I startCell = 2 * cell - UT_Vector3I(1);

			    for (int zOffset = 0; zOffset < 4; ++zOffset)
				for (int yOffset = 0; yOffset < 4; ++yOffset)
				{
				    sourceProbes[yOffset][zOffset].setIndex(startCell[0], startCell[1] + yOffset, startCell[2] + zOffset);
				    
				    for (int xOffset = 0; xOffset < 4; ++xOffset)
				    {
					UT_Vector3I sampleCell = startCell + UT_Vector3I(xOffset, yOffset, zOffset);

					assert(sampleCell[0] >= 0 && sampleCell[1] >= 0 && sampleCell[2] >= 0 &&
						sampleCell[0] < source.getVoxelRes()[0] &&
						sampleCell[1] < source.getVoxelRes()[1] &&
						sampleCell[2] < source.getVoxelRes()[2]);

					sampleValue += restrictionWeights[xOffset] *
							restrictionWeights[yOffset] *
							restrictionWeights[zOffset] *
							SolveReal(sourceProbes[yOffset][zOffset].getValue(xOffset));

#if !defined(NDEBUG)
					auto sourceLabel = sourceCellLabels(sampleCell);
					if (sourceLabel != CellLabels::INTERIOR_CELL && sourceLabel != CellLabels::BOUNDARY_CELL)
					    assert(sourceProbes[yOffset][zOffset].getValue(xOffset) == 0);
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
		    for (; !vit.atEnd(); vit.advance())
		    {
			auto destinationLabel = vit.getValue();
			
			if (destinationLabel == CellLabels::INTERIOR_CELL ||
			    destinationLabel == CellLabels::BOUNDARY_CELL)
			{
			    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
			    
			    UT_Vector3T<SolveReal> samplePoint = .5 * (UT_Vector3T<SolveReal>(cell) + UT_Vector3T<SolveReal>(.5)) - UT_Vector3T<SolveReal>(.5);

			    UT_Vector3I startCell = UT_Vector3I(samplePoint);

			    UT_Vector3T<SolveReal> interpWeight = samplePoint - UT_Vector3T<SolveReal>(startCell);

			    SolveReal sampleValues[2][2][2];

			    for (int zOffset : {0,1})
				for (int yOffset : {0,1})
				{
				    sourceProbes[yOffset][zOffset].setIndex(startCell[0], startCell[1] + yOffset, startCell[2] + zOffset);
				    
				    for (int xOffset : {0,1})
				    {
					sampleValues[xOffset][yOffset][zOffset] = sourceProbes[yOffset][zOffset].getValue(xOffset);

#if !defined(NDEBUG)
					UT_Vector3I testCell = startCell + UT_Vector3I(xOffset, yOffset, zOffset);
					auto sourceLabel = sourceCellLabels(testCell);

					if (sourceLabel != CellLabels::INTERIOR_CELL && sourceLabel != CellLabels::BOUNDARY_CELL)
					    assert(sourceProbes[yOffset][zOffset].getValue(xOffset) == 0);
#endif
				    }
				}
			    
			    destinationProbe.setIndex(vit);
			    
			    // !! Warning !!
			    // The 4x factor in prolongation accounts for the effect of factoring out "dx"
			    // terms from the multigrid operations. This itself is not strictly symmetric to
			    // the restriction operator, but it is symmetric when the implicit "dx" is accounted for.
			    destinationProbe.setValue(destinationProbe.getValue() + 4. * trilerp(sampleValues[0][0][0], sampleValues[1][0][0], sampleValues[0][1][0], sampleValues[1][1][0],
												sampleValues[0][0][1], sampleValues[1][0][1], sampleValues[0][1][1], sampleValues[1][1][1],
												interpWeight[0], interpWeight[1], interpWeight[2]));
			}
		    }
		}
	    }
	});
    }

    template<typename SolveReal, typename StoreReal>
    void
    scaleVector(UT_VoxelArray<StoreReal> &vector,
		const SolveReal scale,
		const UT_VoxelArray<int> &cellLabels)
    {
	assert(vector.getVoxelRes() == cellLabels.getVoxelRes());

	UT_Interrupt *boss = UTgetInterrupt();

        UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    UT_VoxelProbe<StoreReal, true /* read */, true /* write */, true /* test for write */> vectorProbe;
	    vectorProbe.setArray(&vector);
	    
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
			    vectorProbe.setIndex(vit);
			    vectorProbe.setValue(scale * vectorProbe.getValue());
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
    buildExpandedCellLabels(UT_VoxelArray<int> &expandedCellLabels,
			    const UT_VoxelArray<int> &baseCellLabels,
			    const IsExteriorCellFunctor &isExteriorCell,
			    const IsInteriorCellFunctor &isInteriorCell,
			    const IsDirichletCellFunctor &isDirichletCell)
    {
	// Build domain labels with the appropriate padding to apply
	// geometric multigrid directly without a wasteful transfer
	// for each v-cycle.

	// Cap MG levels at 4 voxels in the smallest dimension
	fpreal minLog = std::min(std::log2(fpreal(baseCellLabels.getVoxelRes()[0])),
				    std::log2(fpreal(baseCellLabels.getVoxelRes()[1])));
	minLog = std::min(minLog, std::log2(fpreal(baseCellLabels.getVoxelRes()[2])));

	int mgLevels = ceil(minLog) - std::log2(fpreal(2));

	// Add the necessary exterior cells so that after coarsening to the top level
	// there is still a single layer of exterior cells
	int exteriorPadding = std::pow(2, mgLevels - 1);

	UT_Vector3I expandedResolution = baseCellLabels.getVoxelRes() + 2 * UT_Vector3I(exteriorPadding);

	// Expand the domain to be a power of 2.
	for (int axis : {0,1,2})
	{
	    fpreal logSize = std::log2(fpreal(expandedResolution[axis]));
	    logSize = std::ceil(logSize);

	    expandedResolution[axis] = exint(std::exp2(logSize));
	}
    
	UT_Vector3I exteriorOffset = UT_Vector3I(exteriorPadding);

	expandedCellLabels.size(expandedResolution[0], expandedResolution[1], expandedResolution[2]);
	expandedCellLabels.constant(CellLabels::EXTERIOR_CELL);

	// Build domain cell labels
	UT_Interrupt *boss = UTgetInterrupt();

	// Uncompress internal domain label tiles
	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(expandedCellLabels.numTiles());
	isTileOccupiedList.constant(false);

	UTparallelForEachNumber(baseCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&baseCellLabels);

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() || !isExteriorCell(vit.getValue()))
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (!isExteriorCell(vit.getValue()))
			{
			    UT_Vector3I expandedCell = UT_Vector3I(vit.x(), vit.y(), vit.z()) + exteriorOffset;

			    int tileNumber = expandedCellLabels.indexToLinearTile(expandedCell[0], expandedCell[1], expandedCell[2]);
			    if (!isTileOccupiedList[tileNumber])
				isTileOccupiedList[tileNumber] = true;
			}
		    }
		}
	    }
	});

	uncompressTiles(expandedCellLabels, isTileOccupiedList);

	// Copy initial domain labels to interior domain labels with padding
	UTparallelForEachNumber(baseCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&baseCellLabels);

	    UT_VoxelProbe<int, false /* no read */, true /* write */, true /* test for write */> expandedCellLabelProbe;
	    expandedCellLabelProbe.setArray(&expandedCellLabels);

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() || !isExteriorCell(vit.getValue()))
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			const int baseCellLabel = vit.getValue();
			if (!isExteriorCell(baseCellLabel))
			{
			    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
			    UT_Vector3I expandedCell = cell + exteriorOffset;

			    assert(!expandedCellLabels.getLinearTile(expandedCellLabels.indexToLinearTile(expandedCell[0],
													    expandedCell[1],
													    expandedCell[2]))->isConstant());

			    expandedCellLabelProbe.setIndex(expandedCell[0], expandedCell[1], expandedCell[2]);

			    if (isInteriorCell(baseCellLabel))
				expandedCellLabelProbe.setValue(CellLabels::INTERIOR_CELL);
			    else
			    {
				assert(isDirichletCell(baseCellLabel) && !isExteriorCell(baseCellLabel));
				expandedCellLabelProbe.setValue(CellLabels::DIRICHLET_CELL);
			    }
			}
		    }
		}
	    }
	});

	return std::pair<UT_Vector3I, int>(exteriorOffset, mgLevels);
    }

    template<typename StoreReal>
    void
    buildExpandedBoundaryWeights(UT_VoxelArray<StoreReal> &expandedBoundaryWeights,
				    const UT_VoxelArray<StoreReal> &baseBoundaryWeights,
				    const UT_VoxelArray<int> &expandedCellLabels,
				    const UT_Vector3I &exteriorOffset,
				    const int axis)
    {
	using SIM::FieldUtils::setFieldValue;
	using SIM::FieldUtils::getFieldValue;
	using SIM::FieldUtils::faceToCellMap;

#if !defined(NDEBUG)	
	UT_Vector3I boundaryVoxelRes = expandedBoundaryWeights.getVoxelRes();
	--boundaryVoxelRes[axis];
	assert(boundaryVoxelRes == expandedCellLabels.getVoxelRes());

	assert(expandedBoundaryWeights.getVoxelRes()[0] >= baseBoundaryWeights.getVoxelRes()[0]  + exteriorOffset[0] &&
		expandedBoundaryWeights.getVoxelRes()[1] >= baseBoundaryWeights.getVoxelRes()[1] + exteriorOffset[1] &&
		expandedBoundaryWeights.getVoxelRes()[2] >= baseBoundaryWeights.getVoxelRes()[2] + exteriorOffset[2]);
#endif
	// Make sure weights are empty
	expandedBoundaryWeights.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	UT_Array<bool> isTileOccupiedList;
	isTileOccupiedList.setSize(expandedBoundaryWeights.numTiles());
	isTileOccupiedList.constant(false);

	UTparallelForEachNumber(baseBoundaryWeights.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<StoreReal> vit;
	    vit.setConstArray(&baseBoundaryWeights);

	    if (boss->opInterrupt())
		return;
	 
	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (!vit.isTileConstant() || vit.getValue() > 0)
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() > 0)
			{
			    UT_Vector3I face(vit.x(), vit.y(), vit.z());

			    UT_Vector3I expandedFace = face + exteriorOffset;

			    int tileNumber = expandedBoundaryWeights.indexToLinearTile(expandedFace[0], expandedFace[1], expandedFace[2]);

			    if (!isTileOccupiedList[tileNumber])
				isTileOccupiedList[tileNumber] = true;
			}
		    }
		}
	    }
	});
    
	uncompressTiles(expandedBoundaryWeights, isTileOccupiedList);

	UTparallelForEachNumber(baseBoundaryWeights.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<StoreReal> vit;
	    vit.setConstArray(&baseBoundaryWeights);

	    if (boss->opInterrupt())
		return;

	    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for writes */> expandedWeightsProbe;
	    expandedWeightsProbe.setArray(&expandedBoundaryWeights);
	 
	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (!vit.isTileConstant() || vit.getValue() > 0)
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() > 0)
			{
			    UT_Vector3I face(vit.x(), vit.y(), vit.z());

#if !defined(NDEBUG)
			    UT_Vector3I backwardCell = faceToCellMap(face, axis, 0);
			    UT_Vector3I forwardCell = faceToCellMap(face, axis, 1);

			    assert(expandedCellLabels(backwardCell + exteriorOffset) != CellLabels::EXTERIOR_CELL &&
				    expandedCellLabels(forwardCell + exteriorOffset) != CellLabels::EXTERIOR_CELL);
#endif

			    UT_Vector3I expandedFace = face + exteriorOffset;
			    assert(!expandedBoundaryWeights.getLinearTile(expandedBoundaryWeights.indexToLinearTile(expandedFace[0],
														    expandedFace[1],
														    expandedFace[2]))->isConstant());

			    expandedWeightsProbe.setIndex(face[0] + exteriorOffset[0],
							    face[1] + exteriorOffset[1],
							    face[2] + exteriorOffset[2]);

			    expandedWeightsProbe.setValue(vit.getValue());
			}
		    }
		}
	    }
	});
    }

    template<typename StoreReal>
    void
    setBoundaryCellLabels(UT_VoxelArray<int> &cellLabels,
			    const std::array<UT_VoxelArray<StoreReal>, 3> &boundaryWeights)
    {
	using SIM::FieldUtils::cellToCellMap;
	using SIM::FieldUtils::cellToFaceMap;
	using HDK::GeometricMultigridOperators::CellLabels;

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	    {
		vit.myTileStart = tileNumber;
		vit.myTileEnd = tileNumber + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL)
			{
			    UT_Vector3I cell(vit.x(), vit.y(), vit.z());

			    bool isBoundaryCell = false;

			    for (int axis = 0; axis < 3 & !isBoundaryCell; ++axis)
				for (int direction : {0,1})
				{
				    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < cellLabels.getVoxelRes()[axis]);

				    if (cellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL ||
					cellLabels(adjacentCell) == CellLabels::EXTERIOR_CELL)
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

			    if (isBoundaryCell)
			    {
				assert(!cellLabels.getLinearTile(cellLabels.indexToLinearTile(cell[0], cell[1], cell[2]))->isConstant());
				cellLabels.setValue(cell, CellLabels::BOUNDARY_CELL);
			    }
			}
			else assert(vit.getValue() != CellLabels::BOUNDARY_CELL);
		    }
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

	    for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	    {
		vit.myTileStart = tileNumber;
		vit.myTileEnd = tileNumber + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    break;

		if (!vit.isTileConstant())
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == CellLabels::INTERIOR_CELL ||
			    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    UT_Vector3I cell = UT_Vector3I(vit.x(), vit.y(), vit.z());

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
	});

	uncompressTiles(grid, isTileOccupiedList);
    }

    template<typename StoreReal>
    bool
    unitTestBoundaryCells(const UT_VoxelArray<int> &cellLabels,
			    const std::array<UT_VoxelArray<StoreReal>, 3> *boundaryWeights)
    {
	using SIM::FieldUtils::cellToFaceMap;

	if (boundaryWeights != nullptr)
	{
	    assert( (*boundaryWeights)[0].getVoxelRes()[0] - 1 == cellLabels.getVoxelRes()[0] &&
		    (*boundaryWeights)[0].getVoxelRes()[1]     == cellLabels.getVoxelRes()[1] &&
		    (*boundaryWeights)[0].getVoxelRes()[2]     == cellLabels.getVoxelRes()[2] &&

		    (*boundaryWeights)[1].getVoxelRes()[0]     == cellLabels.getVoxelRes()[0] &&
		    (*boundaryWeights)[1].getVoxelRes()[1] - 1 == cellLabels.getVoxelRes()[1] &&
		    (*boundaryWeights)[1].getVoxelRes()[2]     == cellLabels.getVoxelRes()[2] &&

		    (*boundaryWeights)[2].getVoxelRes()[0]     == cellLabels.getVoxelRes()[0] &&
		    (*boundaryWeights)[2].getVoxelRes()[1]     == cellLabels.getVoxelRes()[1] &&
		    (*boundaryWeights)[2].getVoxelRes()[2] - 1 == cellLabels.getVoxelRes()[2]);
	}

	UT_Interrupt *boss = UTgetInterrupt();

	bool boundaryCellTestPassed = true;

	UTparallelForEachNumber(cellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    if (!boundaryCellTestPassed)
		return;

	    UT_VoxelArrayIterator<int> vit;
	    vit.setConstArray(&cellLabels);

	    for (int i = range.begin(); i != range.end(); ++i)
	    {
		vit.myTileStart = i;
		vit.myTileEnd = i + 1;
		vit.rewind();

		if (boss->opInterrupt())
		    return;
		    
		if (!vit.isTileConstant() ||
		    vit.getValue() == CellLabels::INTERIOR_CELL ||
		    vit.getValue() == CellLabels::BOUNDARY_CELL)
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			UT_Vector3I cell(vit.x(), vit.y(), vit.z());

			if (vit.getValue() == CellLabels::INTERIOR_CELL)
			{
			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I adjacentCell = SIM::FieldUtils::cellToCellMap(cell, axis, direction);

				    if (!(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
					    cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL))
				    {
					boundaryCellTestPassed = false;
					return;
				    }
				}
			}
			else if (vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    bool hasValidBoundary = false;

			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I adjacentCell = SIM::FieldUtils::cellToCellMap(cell, axis, direction);

				    if (!(cellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
					    cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL))
					    hasValidBoundary = true;
				    else if (boundaryWeights != nullptr)
				    {
					UT_Vector3I face = cellToFaceMap(cell, axis, direction);
					
					if ((*boundaryWeights)[axis](face) != 1 && cellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL)
					    hasValidBoundary = true;
				    }
				}

			    if (!hasValidBoundary)
			    {
				boundaryCellTestPassed = false;
				return;
			    }
			}
		    }
		}
	    }
	});

	return boundaryCellTestPassed;
    }

} // HDK::GeometricMultigridOperators

#endif