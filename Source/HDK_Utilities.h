#ifndef HDK_UTILITIES_H
#define HDK_UTILITIES_H

#include <Eigen/Sparse>

#include <SIM/SIM_FieldUtils.h>
#include <SIM/SIM_RawIndexField.h>

#include <UT/UT_ParallelUtil.h>

class SIM_VectorField;
class SIM_ScalarField;

namespace HDK::Utilities
{
    // Standard pressure solver labels
    static constexpr exint UNLABELLED_CELL = -1;
    static constexpr exint LIQUID_CELL = -2;
    static constexpr exint AIR_CELL = -3;

    // Valid face labels
    static constexpr fpreal VALID_FACE = 1;
    static constexpr fpreal INVALID_FACE = 0;

    template<typename SolveReal>
    SYS_FORCE_INLINE
    SolveReal
    computeGhostFluidWeight(SolveReal phi0, SolveReal phi1)
    {
	SolveReal theta = 0;
	if (phi0 < 0)
	{
	    if (phi1 < 0)
		theta = 1;
	    else if (phi1 >= 0)
		theta = phi0 / (phi0 - phi1);
	}
	else if (phi1 < 0)
	    theta = phi1 / (phi1 - phi0);

	return theta;
    }

    exint
    buildLiquidCellIndices(SIM_RawIndexField &liquidCellIndices,
			    const SIM_RawIndexField &liquidCellLabels);

    void
    buildLiquidCellLabels(SIM_RawIndexField &materialCellLabels,
				    const SIM_RawField &liquidSurface,
				    const SIM_RawField &solidSurface,
				    const std::array<SIM_RawField, 3> &cutCellWeights);

    void
    findOccupiedIndexTiles(UT_Array<bool> &isTileOccupiedList,
			    const UT_Array<UT_Vector3I> &indexCellList,
			    const SIM_RawIndexField &indexCellLabels);

    template<typename Grid>
    void
    uncompressTiles(Grid &grid, const UT_Array<bool> &isTileOccupiedList)
    {
	UT_Interrupt *boss = UTgetInterrupt();

	if (boss->opInterrupt())
	    return;

	UTparallelFor(UT_BlockedRange<exint>(0, isTileOccupiedList.size()), [&](const UT_BlockedRange<exint> &range)
	{
	    for (exint tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	    {
		if (!(tileNumber & 127))
		{
		    if (boss->opInterrupt())
			return;
		}

		if (isTileOccupiedList[tileNumber])
		    grid.field()->getLinearTile(tileNumber)->uncompress();
	    }
	});
    }

    // TODO: make temporary list to fill in?
    template<typename ActiveCellFunctor>
    void
    findOccupiedFaceTiles(UT_Array<bool> &isTileOccupiedList,
			    const SIM_RawField &validFaces,
			    const SIM_RawIndexField &activeCellLabels,
			    const ActiveCellFunctor &activeCellFunctor,
			    const int axis)
    {
	using SIM::FieldUtils::cellToFaceMap;

	assert(activeCellLabels.getVoxelRes() == validFaces.getVoxelRes());

	UT_Interrupt *boss = UTgetInterrupt();
	const int tileCount = activeCellLabels.field()->numTiles();
	UTparallelForEachNumber(tileCount, [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIteratorI vit;
	    vit.setConstArray(activeCellLabels.field());

	    UT_VoxelTileIteratorI vitt;

	    for (exint i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

                if (!vit.atEnd())
                {
		    if (boss->opInterrupt())
			return;

		    if (!vit.isTileConstant() || activeCellFunctor(vit.getValue()))
		    {
			vitt.setTile(vit);

			for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			{
			    if (activeCellFunctor(vitt.getValue()))
			    {
				const UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				for (int direction : {0,1})
				{
				    const UT_Vector3I face = cellToFaceMap(cell, axis, direction);

				    int tileNumber = validFaces.field()->indexToLinearTile(face[0], face[1], face[2]);
				    
				    if (!isTileOccupiedList[tileNumber])
					isTileOccupiedList[tileNumber] = true;
				}
			    }
			}
		    }
		}
	    }
	});
    }

    template<typename ActiveCellFunctor>
    void
    classifyValidFaces(SIM_RawField &validFaces,
			const SIM_RawIndexField &activeCellLabels,
			const SIM_RawField &cutCellWeights,
			const ActiveCellFunctor &activeCellFunctor,
			const int axis)
    {
	using namespace SIM::FieldUtils;

	UT_Interrupt *boss = UTgetInterrupt();

	const UT_Vector3I voxelRes = activeCellLabels.getVoxelRes();

	assert(activeCellLabels.getVoxelRes() == validFaces.getVoxelRes() &&
		validFaces.getVoxelRes() == cutCellWeights.getVoxelRes());

	UTparallelForEachNumber(validFaces.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIteratorF vit(validFaces.fieldNC());
	    vit.setCompressOnExit(true);

	    for (exint i = range.begin(); i != range.end(); ++i)
            {
                vit.myTileStart = i;
                vit.myTileEnd = i + 1;
                vit.rewind();

		if (boss->opInterrupt())
		    return;

		if (!vit.isTileConstant())
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			UT_Vector3I face(vit.x(), vit.y(), vit.z());

			if (getFieldValue(cutCellWeights, face) > 0)
			{
			    // If either cell has a valid liquid cell label then this
			    // face will contain a valid velocity value.
			    UT_Vector3I backwardCell = faceToCellMap(face, axis, 0);
			    UT_Vector3I forwardCell = faceToCellMap(face, axis, 1);

			    if (backwardCell[axis] >= 0  && forwardCell[axis] < voxelRes[axis])
			    {
				if (activeCellFunctor(getFieldValue(activeCellLabels, backwardCell)) ||
				    activeCellFunctor(getFieldValue(activeCellLabels, forwardCell)))
				{
				    vit.setValue(HDK::Utilities::VALID_FACE);
				}
			    }
			}
		    }
		}
	    }
	});
    }

    template<typename Vector, typename MatrixVectorMultiplyFunctor, typename PreconditionerFunctor>
    void
    solveConjugateGradient(Vector &solution,
			    const Vector &rhs,
			    const MatrixVectorMultiplyFunctor &matrixVectorMultiplyFunctor,
			    const PreconditionerFunctor &preconditionerFunctor,
			    const double tolerance,
			    const int maxIterations,
			    const bool doProjectNullSpace = false)
    {
	double rhsNorm2 = rhs.squaredNorm();
	if(rhsNorm2 == 0) 
	{
	    solution.setZero();
	    std::cout << "RHS is zero. Nothing to solve" << std::endl;
	    return;
	}

	Vector residual = rhs - matrixVectorMultiplyFunctor(solution); //initial residual

	if (doProjectNullSpace)
	    residual.array() -= residual.sum() / double(residual.rows());

	double threshold = tolerance * tolerance * rhsNorm2;
	double residualNorm2 = residual.squaredNorm();

	if (residualNorm2 < threshold)
	{
	    std::cout << "Residual already below error: " << std::sqrt(residualNorm2 / rhsNorm2) << std::endl;
	    return;
	}

	Vector z(rhs.rows()), tmp(rhs.rows());

	//
	// Build initial search direction from preconditioner
	//

	Vector p = preconditionerFunctor(residual);

	if (doProjectNullSpace)
	    p.array() -= p.sum() / double(p.rows());

	double absNew = residual.dot(p);

	int iteration = 0;
	UT_Interrupt *boss = UTgetInterrupt();

	while (iteration < maxIterations)
	{
   	    tmp = matrixVectorMultiplyFunctor(p);

	    double alpha = absNew / p.dot(tmp);
	    solution += alpha * p;
	    residual -= alpha * tmp;

	    if (doProjectNullSpace)
		residual.array() -= residual.sum() / double(residual.rows());
	
	    residualNorm2 = residual.squaredNorm();
	    if (residualNorm2 < threshold)
		break;
	    else std::cout << "    Residual: " << std::sqrt(residualNorm2) << std::endl;
	    
	    if (boss->opInterrupt())
		return;

	    // Start with the diagonal preconditioner
	    z = preconditionerFunctor(residual);

	    double absOld = absNew;
	    absNew = residual.dot(z);     // update the absolute value of r
	    double beta = absNew / absOld;	  // calculate the Gram-Schmidt value used to create the new search direction
	    p = z + beta * p;			   // update search direction

	    if (doProjectNullSpace)
		p.array() -= p.sum() / double(p.rows());

	    ++iteration;
	}

	std::cout << "Iterations: " << iteration << std::endl;
	std::cout << "Relative L2 Error: " << std::sqrt(residualNorm2 / rhs.squaredNorm()) << std::endl;

	residual = rhs - matrixVectorMultiplyFunctor(solution);
	residualNorm2 = residual.squaredNorm();
	std::cout << "Re-computed Relative L2 Error: " << std::sqrt(residualNorm2 / rhs.squaredNorm()) << std::endl;

	std::cout << "L-infinity Error: " << residual.template lpNorm<Eigen::Infinity>() << std::endl;
    }
} // namespace HDK::Utilities
#endif