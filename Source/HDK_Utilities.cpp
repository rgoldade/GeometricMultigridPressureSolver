#include "HDK_Utilities.h"

namespace HDK::Utilities
{
    bool isCellLiquid(const SIM_RawField &liquidSurface,
			const SIM_RawField &solidSurface,
			const std::array<const SIM_RawField *, 3> &cutCellWeights,
			const UT_Vector3I &cell)
    {
	using SIM::FieldUtils::getFieldValue;
	using SIM::FieldUtils::cellToCellMap;
	using SIM::FieldUtils::cellToFaceMap;

	if (getFieldValue(liquidSurface, cell) <= 0.)
	    return true;

	// If the cell is not inside the liquid, it could still be labelled a valid liquid cell if
	// 1. the cell is inside a solid
	// 2. the adjacent cell is inside liquid
	// 3. the face weight between the two cells is greater than zero

	UT_Vector3 pos;
	liquidSurface.indexToPos(cell[0], cell[1], cell[2], pos);

	if (solidSurface.getValue(pos) >= 0)
	{
	    for (int axis : {0,1,2})
		for (int direction : {0,1})
		{
		    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
		    
		    if (getFieldValue(*cutCellWeights[axis], face) > 0)
		    {
			UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

			if (adjacentCell[axis] < 0 || adjacentCell[axis] >= liquidSurface.getVoxelRes()[axis])
			    continue;

			if (getFieldValue(liquidSurface, adjacentCell) <= 0)
			    return true;
		    }
    		}
	}

	return false;
    }

    exint
    buildLiquidCellIndices(SIM_RawIndexField &liquidCellIndices,
			    const SIM_RawIndexField &materialCellLabels)
    {
	using SIM::FieldUtils::setFieldValue;

	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());

	UT_VoxelTileIteratorI vitt;

	exint liquidCellCount = 0;

	// Build liquid cell indices
	UT_Interrupt *boss = UTgetInterrupt();
	for (vit.rewind(); !vit.atEnd(); vit.advanceTile())
	{
	    if (boss->opInterrupt())
		break;

	    if (!vit.isTileConstant() || vit.getValue() == FreeSurfaceMaterialLabels::LIQUID_CELL)
	    {
		vitt.setTile(vit);

		for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		{
		    if (vitt.getValue() == FreeSurfaceMaterialLabels::LIQUID_CELL)
		    {
			UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
			setFieldValue(liquidCellIndices, cell, liquidCellCount++);
		    }
		}
	    }
	}

	return liquidCellCount;
    }

    void
    buildMaterialCellLabels(SIM_RawIndexField &materialCellLabels,
			    const SIM_RawField &liquidSurface,
			    const SIM_RawField &solidSurface,
			    const std::array<const SIM_RawField *, 3> &cutCellWeights)
    {
	using SIM::FieldUtils::getFieldValue;
	using SIM::FieldUtils::setFieldValue;
	using SIM::FieldUtils::cellToFaceMap;

	assert(materialCellLabels.isAligned(&liquidSurface));

	materialCellLabels.makeConstant(FreeSurfaceMaterialLabels::SOLID_CELL);

	UT_Interrupt *boss = UTgetInterrupt();

	UTparallelForEachNumber(liquidSurface.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIteratorF vit;
	    vit.setConstArray(liquidSurface.field());

	    UT_VoxelTileIteratorF vitt;

	    if (boss->opInterrupt())
		return;
	 
	    for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	    {
		vit.myTileStart = tileNumber;
		vit.myTileEnd = tileNumber + 1;
		vit.rewind();

		if (!vit.atEnd())
		{
		    vitt.setTile(vit);
		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			bool isInFluid = false;

			UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I face = cellToFaceMap(cell, axis, direction);

				if (getFieldValue(*cutCellWeights[axis], face) > 0)
				    isInFluid = true;
			    }

			if (isInFluid)
			{
			    if (isCellLiquid(liquidSurface, solidSurface, cutCellWeights, cell))
				setFieldValue(materialCellLabels, cell, FreeSurfaceMaterialLabels::LIQUID_CELL);
			    else
				setFieldValue(materialCellLabels, cell, FreeSurfaceMaterialLabels::AIR_CELL);
			}
		    }
		}
	    }
	});

	materialCellLabels.fieldNC()->collapseAllTiles();
    }

    void findOccupiedIndexTiles(UT_Array<bool> &isTileOccupiedList,
				const UT_Array<UT_Vector3I> &indexCellList,
				const SIM_RawIndexField &indexCellLabels)
    {
	UT_Interrupt *boss = UTgetInterrupt();

	const exint tileSize = isTileOccupiedList.entries();

	UT_Array<bool> localIsTileOccupiedList;
	localIsTileOccupiedList.setSize(tileSize);
	localIsTileOccupiedList.constant(false);
	
	if (boss->opInterrupt())
	    return;

	const exint elementSize = indexCellList.entries();
	UTparallelFor(UT_BlockedRange<exint>(0, elementSize), [&](const UT_BlockedRange<exint> &range)
	{
	    for (exint i = range.begin(); i != range.end(); ++i)
	    {
		if (!(i & 127))
		{
		    if (boss->opInterrupt())
			break;
		}

		UT_Vector3I cell = indexCellList[i];

		int tileNumber = indexCellLabels.field()->indexToLinearTile(cell[0], cell[1], cell[2]);

		localIsTileOccupiedList[tileNumber] = true;
	    }
	});

	for (exint tileNumber = 0; tileNumber < tileSize; ++tileNumber)
	{
	    if (localIsTileOccupiedList[tileNumber])
		isTileOccupiedList[tileNumber] = true;
	}
    }
} // namespace HDK::Utilities