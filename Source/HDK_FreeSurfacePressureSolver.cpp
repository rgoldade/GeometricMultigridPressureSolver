#include "HDK_FreeSurfacePressureSolver.h"

#include <PRM/PRM_Include.h>

#include <SIM/SIM_DopDescription.h>
#include <SIM/SIM_FieldUtils.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_PRMShared.h>

#include <UT/UT_DSOVersion.h>
#include <UT/UT_PerfMonAutoEvent.h>
#include <UT/UT_ThreadedAlgorithm.h>

#include "HDK_Utilities.h"

void
initializeSIM(void *)
{
   IMPLEMENT_DATAFACTORY(HDK_FreeSurfacePressureSolver);
}

// Standard constructor, note that BaseClass was crated by the
// DECLARE_DATAFACTORY and provides an easy way to chain through
// the class hierarchy.
HDK_FreeSurfacePressureSolver::HDK_FreeSurfacePressureSolver(const SIM_DataFactory *factory)
    : BaseClass(factory)
{
}

HDK_FreeSurfacePressureSolver::~HDK_FreeSurfacePressureSolver()
{
}

const SIM_DopDescription*
HDK_FreeSurfacePressureSolver::getDopDescription()
{
    static PRM_Name	theSurfaceName(GAS_NAME_SURFACE, "Surface Field");
    static PRM_Default 	theSurfaceDefault(0,"surface");

    static PRM_Name	theVelocityName(GAS_NAME_VELOCITY, "Velocity Field");
    static PRM_Default	theVelocityDefault(0, "vel");

    static PRM_Name    theSolidSurfaceName(GAS_NAME_COLLISION, "Solid Field");
    static PRM_Default theSolidSurfaceDefault(0, "collision");

    static PRM_Name	theSolidVelocityName(GAS_NAME_COLLISIONVELOCITY, "Solid Velocity Field");
    static PRM_Default  theSolidVelocityDefault(0, "collisionvel");

    static PRM_Name 	theCutCellWeightsName("cutCellWeights", "Cut-cell Weights Field");
    static PRM_Default  theCutCellWeightsDefault(0, "collisionweights");

    static PRM_Name 	thePressureName(GAS_NAME_PRESSURE, "Pressure");
    static PRM_Default 	thePressureNameDefault(0, "pressure");

    static PRM_Name    theUseOldPressureName("useOldPressure", "Use old pressure as an initial guess");

    static PRM_Name 	theDensityName(GAS_NAME_DENSITY, "Liquid Density Field");
    static PRM_Default 	theDensityNameDefault(0, "massdensity");

    static PRM_Name 	theValidFacesName("validFaces", "Valid Faces Field");

    static PRM_Name 	theSolverToleranceName(SIM_NAME_TOLERANCE, "Solver Tolerance");
    static PRM_Default 	theSolverToleranceDefault(1e-5);

    static PRM_Name 	theMaxIterationsName("maxIterations", "Max Solver Iterations");
    static PRM_Default 	theMaxIterationsDefault(2500);

    static PRM_Template	theTemplates[] =
    {
        PRM_Template(PRM_STRING, 1, &theSurfaceName, &theSurfaceDefault),

    	PRM_Template(PRM_STRING, 1, &theVelocityName, &theVelocityDefault),

    	PRM_Template(PRM_STRING, 1, &theSolidSurfaceName, &theSolidSurfaceDefault),

    	PRM_Template(PRM_STRING, 1, &theSolidVelocityName, &theSolidVelocityDefault),

    	PRM_Template(PRM_STRING, 1, &theCutCellWeightsName, &theCutCellWeightsDefault),

        PRM_Template(PRM_STRING, 1, &thePressureName, &thePressureNameDefault),

	PRM_Template(PRM_TOGGLE, 1, &theUseOldPressureName, PRMoneDefaults),

        PRM_Template(PRM_STRING, 1, &theDensityName, &theDensityNameDefault),

        PRM_Template(PRM_STRING, 1, &theValidFacesName),

    	PRM_Template(PRM_FLT, 1, &theSolverToleranceName, &theSolverToleranceDefault),

    	PRM_Template(PRM_INT, 1, &theMaxIterationsName, &theMaxIterationsDefault),

    	PRM_Template()
    };

    static SIM_DopDescription theDopDescription(true,
						"HDK_FreeSurfacePressureSolver",
						"HDK Free Surface Pressure Solver",
						"$OS",
						classname(),
						theTemplates);

    setGasDescription(theDopDescription);

    return &theDopDescription;
}

bool
HDK_FreeSurfacePressureSolver::solveGasSubclass(SIM_Engine &engine,
						SIM_Object *obj,
						SIM_Time time,
						SIM_Time timestep)
{
    const SIM_VectorField *solidVelocity = getConstVectorField(obj, GAS_NAME_COLLISIONVELOCITY);

    // Load liquid velocity

    SIM_VectorField *velocity = getVectorField(obj, GAS_NAME_VELOCITY);

    if (velocity == nullptr)
    {
        addError(obj, SIM_MESSAGE, "Velocity field missing", UT_ERROR_WARNING);
        return false;
    }
    else if (!velocity->isFaceSampled())
    {
        addError(obj, SIM_MESSAGE, "Velocity field must be a staggered grid", UT_ERROR_WARNING);
        return false;
    }

    // Load cut-cell weights

    const SIM_VectorField *cutCellWeightsField = getConstVectorField(obj, "cutCellWeights");
    
    if (cutCellWeightsField == nullptr)
    {
        addError(obj, SIM_MESSAGE, "Cut-cell weights field missing", UT_ERROR_WARNING);
        return false;
    }
    else if (!cutCellWeightsField->isAligned(velocity))
    {
        addError(obj, SIM_MESSAGE, "Cut-cell weights must align with velocity samples", UT_ERROR_WARNING);
        return false;
    }
    
    std::array<SIM_RawField, 3> cutCellWeights;
    for (int axis : {0, 1, 2})
	cutCellWeights[axis] = *(cutCellWeightsField->getField(axis));

    // Load valid fluid faces

    SIM_VectorField *validFaces = getVectorField(obj, "validFaces");

    if (validFaces == nullptr)
    {
        addError(obj, SIM_MESSAGE, "No 'valid' field found", UT_ERROR_ABORT);
        return false;
    }
    else if (!validFaces->isAligned(velocity))
    {
        addError(obj, SIM_MESSAGE, "Valid field sampling needs to match velocity field", UT_ERROR_ABORT);
        return false;
    }

    // Load liquid SDF

    const SIM_ScalarField *liquidSurfaceField = getConstScalarField(obj, GAS_NAME_SURFACE);

    if (liquidSurfaceField == nullptr)
    {
        addError(obj, SIM_MESSAGE, "Surface field is missing. There is nothing to represent the liquid", UT_ERROR_WARNING);
        return false;
    }

    const SIM_RawField &liquidSurface = *liquidSurfaceField->getField();

    // Load liquid pressure

    SIM_ScalarField *pressureField = getScalarField(obj, GAS_NAME_PRESSURE, true);
    SIM_RawField localPressure, *pressure;

    const UT_Vector3I voxelRes = velocity->getTotalVoxelRes();

    if (pressureField == nullptr)
    {
	localPressure.init(SIM_SAMPLE_CENTER,
			    velocity->getOrig(),
			    velocity->getSize(),
			    voxelRes[0], voxelRes[1], voxelRes[2]);
	localPressure.makeConstant(0);
	pressure = &localPressure;
    }
    else
    {
	pressureField->matchField(liquidSurfaceField);
	pressure = pressureField->getField();
    }

    // Load solid SDF

    const fpreal dx = velocity->getVoxelSize().maxComponent();

    const SIM_ScalarField *solidSurfaceField = getConstScalarField(obj, GAS_NAME_COLLISION);
    const SIM_RawField *solidSurface;
    SIM_RawField localSolidSurface;

    if (solidSurfaceField == nullptr)
    {
        // Treat as all fluid.
        localSolidSurface.init(SIM_SAMPLE_CENTER,
				velocity->getOrig(),
				velocity->getSize(),
				voxelRes[0], voxelRes[1], voxelRes[2]);

        // Solid level set convention in Houdini is negative outside and positive inside.
        localSolidSurface.makeConstant(-10. * dx);
        solidSurface = &localSolidSurface;
    }
    else
	solidSurface = solidSurfaceField->getField();

    // Load liquid density

    const SIM_ScalarField *liquidDensityField = getConstScalarField(obj, GAS_NAME_DENSITY);

    if (liquidDensityField == nullptr)
    {
        addError(obj, SIM_MESSAGE, "There is no liquid density to simulate with", UT_ERROR_WARNING);
        return false;
    }

    const SIM_RawField &liquidDensity = *liquidDensityField->getField();

    if (!liquidDensity.isAligned(&liquidSurface))
    {
        addError(obj, SIM_MESSAGE, "Density must align with the surface volume", UT_ERROR_WARNING);
        return false;
    }

    fpreal32 constantLiquidDensity = 0.;
    if (!liquidDensity.field()->isConstant(&constantLiquidDensity))
    {
	addError(obj, SIM_MESSAGE, "Variable density is not currently supported", UT_ERROR_WARNING);
        return false;
    }

    ////////////////////////////////////////////
    //
    // Build a mapping of solvable pressure cells
    // to rows in the matrix.
    //
    ////////////////////////////////////////////

    std::cout << "//\n//\n// Starting free surface pressure solver\n//\n//" << std::endl;

    exint liquidCellCount = 0;
    SIM_RawIndexField liquidCellIndices;
    SIM_RawIndexField materialCellLabels;

    {
    	UT_PerfMonAutoSolveEvent event(this, "Build liquid cell labels");

    	std::cout << "\n// Build liquid cell labels" << std::endl;

    	liquidCellIndices.match(liquidSurface);
	liquidCellIndices.makeConstant(HDK::Utilities::UNLABELLED_CELL);

    	materialCellLabels.match(liquidSurface);
	materialCellLabels.makeConstant(HDK::Utilities::UNLABELLED_CELL);

	HDK::Utilities::buildLiquidCellLabels(materialCellLabels,
						liquidSurface,
						*solidSurface,
						cutCellWeights);

	liquidCellCount = HDK::Utilities::buildLiquidCellIndices(liquidCellIndices,
								    materialCellLabels);

    	UT_WorkBuffer extrainfo;
	extrainfo.sprintf("liquid DOFs=%d", int(liquidCellCount));
	event.setExtraInfo(extrainfo.buffer());
    }

    if (liquidCellCount == 0)
    {
        addError(obj, SIM_MESSAGE, "No liquid cells found", UT_ERROR_WARNING);
        return false;
    }

    ////////////////////////////////////////////
    //
    // Build valid faces to indicate active velocity elements
    //
    ////////////////////////////////////////////

    {
    	std::cout << "\n// Build valid flags" << std::endl;
    	UT_PerfMonAutoSolveEvent event(this, "Build valid flags");

    	buildValidFaces(*validFaces,
			materialCellLabels,
			cutCellWeights);
    }

    ////////////////////////////////////////////
    //
    // Build right-hand-side for liquid degrees of freedom using the 
    // cut-cell method to account for moving solids.
    //
    ////////////////////////////////////////////

    Vector rhsVector = Vector::Zero(liquidCellCount);

    {
    	std::cout << "\n// Build liquid RHS" << std::endl;
    	UT_PerfMonAutoSolveEvent event(this, "Build liquid right-hand side");

    	buildRHS(rhsVector,
		    liquidCellIndices,
		    materialCellLabels,
		    *velocity,
		    solidVelocity,
		    *validFaces,
		    cutCellWeights);
    }

    ////////////////////////////////////////////
    //
    // Build matrix rows for liquid degrees of freedom.
    // Add non-zeros for entries for adjacent air regions.
    //
    ////////////////////////////////////////////

    std::vector<Eigen::Triplet<SolveReal>> poissonElements;
    
    const int threadCount = UT_Thread::getNumProcessors();
    {
        std::cout << "\n// Build liquid matrix rows" << std::endl;
        UT_PerfMonAutoSolveEvent event(this, "Build liquid matrix rows");

	std::vector<std::vector<Eigen::Triplet<SolveReal>>> parallelPoissonElements(threadCount);

    	buildPoissonRows(parallelPoissonElements,
			    liquidCellIndices,
			    materialCellLabels,
			    *validFaces,
			    liquidSurface,
			    cutCellWeights);

	exint listSize = 0;
	for (int thread = 0; thread < threadCount; ++thread)
	    listSize += parallelPoissonElements[thread].size();

	poissonElements.reserve(listSize);

    	for (int thread = 0; thread < threadCount; ++thread)
    	    poissonElements.insert(poissonElements.end(), parallelPoissonElements[thread].begin(), parallelPoissonElements[thread].end());
    }
    
    ////////////////////////////////////////////
    //
    // Solve for pressure.
    //
    ////////////////////////////////////////////

    Vector solutionVector = Vector::Zero(liquidCellCount);

    if (getUseOldPressure())
    {
	std::cout << "\n// Apply warm start pressure" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Apply warm start pressure");

	applyOldPressure(solutionVector,
			    *pressure,
			    liquidCellIndices);
    }

    {
	std::cout << "\n// Solve liquid system" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Solve linear system");

	Eigen::SparseMatrix<SolveReal, Eigen::RowMajor> poissonMatrix(liquidCellCount, liquidCellCount);
	poissonMatrix.setFromTriplets(poissonElements.begin(), poissonElements.end());
	poissonMatrix.makeCompressed();
	
#if !defined(NDEBUG)

	std::cout << "\n    Checking symmetry of system" << std::endl;
	for (int k = 0; k < poissonMatrix.outerSize(); ++k)
	    for (typename Eigen::SparseMatrix<SolveReal, Eigen::RowMajor>::InnerIterator it(poissonMatrix, k); it; ++it)
	    {
		if (!SYSisEqual(poissonMatrix.coeff(it.row(), it.col()), poissonMatrix.coeff(it.col(), it.row())))
		{
		    std::cout << "Value at row " << it.row() << ", col " << it.col() << " is " << poissonMatrix.coeff(it.row(), it.col()) << std::endl;
		    std::cout << "Value at row " << it.col() << ", col " << it.row() << " is " << poissonMatrix.coeff(it.col(), it.row()) << std::endl;
		    assert(false);
		}
	    }
    
	// Check that it's finite
	std::cout << "    Check for NaNs\n" << std::endl;
	for (int k = 0; k < poissonMatrix.outerSize(); ++k)
	    for (typename Eigen::SparseMatrix<SolveReal, Eigen::RowMajor>::InnerIterator it(poissonMatrix, k); it; ++it)
	    {
		if (!std::isfinite(it.value()))
		    assert(false);
	    }
#endif

	Eigen::ConjugateGradient<Eigen::SparseMatrix<SolveReal, Eigen::RowMajor>, Eigen::Lower | Eigen::Upper> solver;
	solver.setTolerance(getSolverTolerance());
	solver.compute(poissonMatrix);

	solutionVector = solver.solveWithGuess(rhsVector, solutionVector);
	
	Vector residualVector = rhsVector - poissonMatrix * solutionVector;

	std::cout << "Solver iterations: " << solver.iterations() << std::endl;
	std::cout << "Solver error: " << solver.error() << std::endl;
	std::cout << " Recomputed relative residual: " << std::sqrt(residualVector.squaredNorm() / rhsVector.squaredNorm()) << std::endl;
    }

    ////////////////////////////////////////////
    //
    // Apply pressure gradient to the velocity field
    //
    ////////////////////////////////////////////

    {
	std::cout << "\n// Apply solution to pressure" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Apply solution to pressure");
	
	pressure->makeConstant(0);

	// Apply solution from liquid cells.
	applySolutionToPressure(*pressure, liquidCellIndices, solutionVector);
    }

    {
	std::cout << "\n// Update velocity from pressure gradient" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Update velocity from pressure gradient");

	for (int axis : {0,1,2})
	{
	    applyPressureGradient(*velocity->getField(axis),
				    *validFaces->getField(axis),
				    *pressure,
				    liquidSurface,
				    cutCellWeights[axis],
				    liquidCellIndices,
				    axis);
	}
    }

    {
	// Check divergence
	UT_VoxelArray<SolveReal> debugDivergenceGrid;
	debugDivergenceGrid.size(materialCellLabels.getVoxelRes()[0], materialCellLabels.getVoxelRes()[1], materialCellLabels.getVoxelRes()[2]);
	debugDivergenceGrid.constant(0);

	buildDebugDivergence(debugDivergenceGrid,
				materialCellLabels,
				*velocity,
				solidVelocity,
				*validFaces,
				cutCellWeights);

	UT_Array<SolveReal> tiledMaxDivergenceList;
	tiledMaxDivergenceList.setSize(materialCellLabels.field()->numTiles());
	tiledMaxDivergenceList.constant(0);

	UT_Array<SolveReal> tiledSumDivergenceList;
	tiledSumDivergenceList.setSize(materialCellLabels.field()->numTiles());
	tiledSumDivergenceList.constant(0);

	buildMaxAndSumGrid(tiledMaxDivergenceList,
			    tiledSumDivergenceList,
			    debugDivergenceGrid,
			    materialCellLabels);

	SolveReal maxDivergence = 0;
	SolveReal sumDivergence = 0;
	for (int tile = 0; tile < materialCellLabels.field()->numTiles(); ++tile)
	{
	    if (fabs(tiledMaxDivergenceList[tile]) > fabs(maxDivergence))
		maxDivergence = tiledMaxDivergenceList[tile];
	    sumDivergence += tiledSumDivergenceList[tile];
	}

	std::cout << "  L-infinity divergence: " << maxDivergence << std::endl;
	std::cout << "  Sum divergence: " << sumDivergence << std::endl;
    }


    pressureField->pubHandleModification();
    velocity->pubHandleModification();
    validFaces->pubHandleModification();

    return true;
}

void
HDK_FreeSurfacePressureSolver::buildRHS(Vector &rhsVector,
					const SIM_RawIndexField &liquidCellIndices,
					const SIM_RawIndexField &materialCellLabels,
					const SIM_VectorField &velocity,
					const SIM_VectorField *solidVelocity,
					const SIM_VectorField &validFaces,
					const std::array<SIM_RawField, 3> &cutCellWeights) const
{
    using SIM::FieldUtils::cellToFaceMap;
    using SIM::FieldUtils::getFieldValue;

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(liquidCellIndices.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(liquidCellIndices.field());

	UT_VoxelTileIteratorI vitt;

	if (boss->opInterrupt())
	    return;

	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant())
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			exint liquidIndex = vitt.getValue();
			if (liquidIndex >= 0)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			    assert(getFieldValue(materialCellLabels, cell) == HDK::Utilities::LIQUID_CELL);

			    fpreal divergence = 0;

			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I face = cellToFaceMap(cell, axis, direction);

				    fpreal sign = (direction % 2 == 0) ? 1. : -1.;
				    fpreal weight = getFieldValue(cutCellWeights[axis], face);

				    if (weight > 0)
				    {
					divergence += sign * weight * getFieldValue(*velocity.getField(axis), face);
					assert(getFieldValue(*validFaces.getField(axis), face) == HDK::Utilities::VALID_FACE);
				    }
				    if (solidVelocity != nullptr && weight < 1)
				    {
					UT_Vector3 point;
					velocity.getField(axis)->indexToPos(face[0], face[1], face[2], point);

					divergence += sign * (1. - weight) * solidVelocity->getField(axis)->getValue(point);
				    }
				}

			    rhsVector(liquidIndex) = divergence;
			}
		    }
		}
	    }
	}
    });  
}

void
HDK_FreeSurfacePressureSolver::buildPoissonRows(std::vector<std::vector<Eigen::Triplet<SolveReal>>> &parallelPoissonElements,
						const SIM_RawIndexField &liquidCellIndices,
						const SIM_RawIndexField &materialCellLabels,
						const SIM_VectorField &validFaces,
						const SIM_RawField &liquidSurface,
						const std::array<SIM_RawField, 3> &cutCellWeights) const
{
    using SIM::FieldUtils::cellToFaceMap;
    using SIM::FieldUtils::cellToCellMap;
    using SIM::FieldUtils::getFieldValue;

    UT_ThreadedAlgorithm buildPoissonSystemAlgorithm;
    buildPoissonSystemAlgorithm.run([&](const UT_JobInfo &info)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(liquidCellIndices.field());
	vit.splitByTile(info);

	UT_VoxelTileIteratorI vitt;

	UT_Interrupt *boss = UTgetInterrupt();

	UT_Vector3I voxelRes = liquidCellIndices.getVoxelRes();

	std::vector<Eigen::Triplet<SolveReal>> &localPoissonElements = parallelPoissonElements[info.job()];

	for (vit.rewind(); !vit.atEnd(); vit.advanceTile())
	{
	    if (boss->opInterrupt())
		break;

	    if (!vit.isTileConstant())
	    {
		vitt.setTile(vit);

		for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		{
		    exint liquidIndex = vitt.getValue();
		    if (liquidIndex >= 0)
		    {
			UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			assert(getFieldValue(materialCellLabels, cell) == HDK::Utilities::LIQUID_CELL);

			// Build non-zeros for each liquid voxel row
			fpreal poissonDiagonal = 0.;
			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				if (adjacentCell[axis] < 0 || adjacentCell[axis] >= voxelRes[axis])
				    continue;

				UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				fpreal weight = getFieldValue(cutCellWeights[axis], face);

				if (weight > 0)
				{
				    assert(getFieldValue(*validFaces.getField(axis), face) == HDK::Utilities::VALID_FACE);
				    assert(weight > 0);
				
				    exint adjacentLiquidIndex = getFieldValue(liquidCellIndices, adjacentCell);

				    if (adjacentLiquidIndex >= 0)
				    {
					assert(getFieldValue(materialCellLabels, adjacentCell) == HDK::Utilities::LIQUID_CELL);
					poissonDiagonal += weight;
					localPoissonElements.push_back(Eigen::Triplet<SolveReal>(liquidIndex, adjacentLiquidIndex, -weight));
				    }
				    else
				    {
					fpreal phi0 = getFieldValue(liquidSurface, cell);
					fpreal phi1 = getFieldValue(liquidSurface, adjacentCell);

					assert(phi1 > 0);
					assert(getFieldValue(materialCellLabels, adjacentCell) == HDK::Utilities::AIR_CELL);

					fpreal theta = HDK::Utilities::computeGhostFluidWeight(phi0, phi1);
					theta = SYSclamp(theta, .01, 1.);

					poissonDiagonal += weight / theta;
				    }
				}
				else
				    assert(getFieldValue(*validFaces.getField(axis), face) == HDK::Utilities::INVALID_FACE);
			    }

			assert(poissonDiagonal > 0.);

			localPoissonElements.push_back(Eigen::Triplet<SolveReal>(liquidIndex, liquidIndex, poissonDiagonal));
		    }
		}
	    }
	}

	return 0;
    });
}

void
HDK_FreeSurfacePressureSolver::applyOldPressure(Vector &solutionVector,
						const SIM_RawField &pressure,
						const SIM_RawIndexField &liquidCellIndices) const
{
    using SIM::FieldUtils::getFieldValue;

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(liquidCellIndices.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(liquidCellIndices.field());

	UT_VoxelTileIteratorI vitt;

	if (boss->opInterrupt())
	    return;
	 
	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant())
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			exint liquidIndex = vitt.getValue();

			if (liquidIndex >= 0)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
			    solutionVector(liquidIndex) = getFieldValue(pressure, cell);
			}
		    }
		}
	    }
	}
    });
}

void
HDK_FreeSurfacePressureSolver::buildMGBoundaryWeights(SIM_RawField &boundaryWeights,
							const SIM_RawField &validFaces,
							const SIM_RawField &liquidSurface,
							const SIM_RawIndexField &liquidCellIndices,
							const int axis) const
{
    using SIM::FieldUtils::setFieldValue;
    using SIM::FieldUtils::getFieldValue;
    using SIM::FieldUtils::faceToCellMap;

    UT_Interrupt *boss = UTgetInterrupt();

    UT_Vector3I voxelRes = liquidCellIndices.getVoxelRes();

    UTparallelForEachNumber(validFaces.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorF vit;
	vit.setConstArray(validFaces.field());

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
		if (!vit.isTileConstant() || vit.getValue() == HDK::Utilities::VALID_FACE)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {	      
			if (vitt.getValue() == HDK::Utilities::VALID_FACE)
			{
			    UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());

			    UT_Vector3I backwardCell = faceToCellMap(face, axis, 0);
			    UT_Vector3I forwardCell = faceToCellMap(face, axis, 1);

			    if (backwardCell[axis] < 0 || forwardCell[axis] >= voxelRes[axis])
			    {
				setFieldValue(boundaryWeights, face, 0);
				continue;
			    }

			    exint backwardIndex = getFieldValue(liquidCellIndices, backwardCell);
			    exint forwardIndex = getFieldValue(liquidCellIndices, forwardCell);

			    assert(backwardIndex >= 0 || forwardIndex >= 0);
			    assert(getFieldValue(boundaryWeights, face) > 0);

			    if (backwardIndex == HDK::Utilities::UNLABELLED_CELL ||
				forwardIndex == HDK::Utilities::UNLABELLED_CELL)
			    {
				fpreal backwardPhi = getFieldValue(liquidSurface, backwardCell);
				fpreal forwardPhi = getFieldValue(liquidSurface, forwardCell);

				fpreal theta = HDK::Utilities::computeGhostFluidWeight(backwardPhi, forwardPhi);
				theta = SYSclamp(theta, .01, 1.);

				setFieldValue(boundaryWeights, face, getFieldValue(boundaryWeights, face) / theta);
			    }
			}
		    }
		}
	    }
	}
    });
}

void
HDK_FreeSurfacePressureSolver::applySolutionToPressure(SIM_RawField &pressure,
							const SIM_RawIndexField &liquidCellIndices,
							const Vector &solutionVector) const
{
    using SIM::FieldUtils::setFieldValue;

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(liquidCellIndices.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(liquidCellIndices.field());

	UT_VoxelTileIteratorI vitt;

	if (boss->opInterrupt())
	    return;
	 
	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant())
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			exint liquidIndex = vitt.getValue();

			if (liquidIndex >= 0)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
			    setFieldValue(pressure, cell, solutionVector(liquidIndex));
			}
		    }
		}
	    }
	}
    });
}

void
HDK_FreeSurfacePressureSolver::buildValidFaces(SIM_VectorField &validFaces,
						const SIM_RawIndexField &materialCellLabels,
						const std::array<SIM_RawField, 3> &cutCellWeights) const
{
    UT_Array<bool> isTileOccupiedList;
    for (int axis : {0,1,2})
    {
	validFaces.getField(axis)->makeConstant(HDK::Utilities::INVALID_FACE);

	isTileOccupiedList.clear();
	isTileOccupiedList.setSize(validFaces.getField(axis)->field()->numTiles());
	isTileOccupiedList.constant(false);

	HDK::Utilities::findOccupiedFaceTiles(isTileOccupiedList,
						*validFaces.getField(axis),
						materialCellLabels,
						[](const exint label){ return label == HDK::Utilities::LIQUID_CELL; },
						axis);

	HDK::Utilities::uncompressTiles(*validFaces.getField(axis), isTileOccupiedList);

	HDK::Utilities::classifyValidFaces(*validFaces.getField(axis),
					    materialCellLabels,
					    cutCellWeights[axis],
					    [](const exint label){ return label == HDK::Utilities::LIQUID_CELL; },
					    axis);
    }
}

void
HDK_FreeSurfacePressureSolver::applyPressureGradient(SIM_RawField &velocity,
							const SIM_RawField &validFaces,
							const SIM_RawField &pressure,
							const SIM_RawField &liquidSurface,
							const SIM_RawField &cutCellWeights,
							const SIM_RawIndexField &liquidCellIndices,
							const int axis) const
{
    using SIM::FieldUtils::setFieldValue;
    using SIM::FieldUtils::getFieldValue;
    using SIM::FieldUtils::faceToCellMap;

    UT_Interrupt *boss = UTgetInterrupt();

    UT_Vector3I voxelRes = pressure.getVoxelRes();

    UTparallelForEachNumber(validFaces.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorF vit;
	vit.setConstArray(validFaces.field());

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
		if (!vit.isTileConstant() || vit.getValue() == HDK::Utilities::VALID_FACE)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {	      
			if (vitt.getValue() == HDK::Utilities::VALID_FACE)
			{
			    UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());

			    UT_Vector3I backwardCell = faceToCellMap(face, axis, 0);
			    UT_Vector3I forwardCell = faceToCellMap(face, axis, 1);

			    if (backwardCell[axis] < 0 || forwardCell[axis] >= voxelRes[axis])
				continue;

			    exint backwardIndex = getFieldValue(liquidCellIndices, backwardCell);
			    exint forwardIndex = getFieldValue(liquidCellIndices, forwardCell);

			    assert(backwardIndex >= 0 || forwardIndex >= 0);
			    assert(getFieldValue(cutCellWeights, face) > 0);

			    fpreal gradient = getFieldValue(pressure, forwardCell) - getFieldValue(pressure, backwardCell);

			    if (backwardIndex == HDK::Utilities::UNLABELLED_CELL ||
				forwardIndex == HDK::Utilities::UNLABELLED_CELL)
			    {
				fpreal backwardPhi = getFieldValue(liquidSurface, backwardCell);
				fpreal forwardPhi = getFieldValue(liquidSurface, forwardCell);

				fpreal theta = HDK::Utilities::computeGhostFluidWeight(backwardPhi, forwardPhi);
				theta = SYSclamp(theta, .01, 1.);

				gradient /= theta;
			    }

			    setFieldValue(velocity, face, getFieldValue(velocity, face) - gradient);
			}
		    }
		}
	    }
	}
    });
}

void
HDK_FreeSurfacePressureSolver::buildDebugDivergence(UT_VoxelArray<SolveReal> &debugDivergenceGrid,
						    const SIM_RawIndexField &materialCellLabels,
						    const SIM_VectorField &velocity,
						    const SIM_VectorField *solidVelocity,
						    const SIM_VectorField &validFaces,
						    const std::array<SIM_RawField, 3> &cutCellWeights) const
{
    using SIM::FieldUtils::cellToFaceMap;
    using SIM::FieldUtils::getFieldValue;

    // Pre-expand all tiles that correspond to INTERIOR or BOUNDARY domain labels
    assert(debugDivergenceGrid.getVoxelRes() == materialCellLabels.getVoxelRes());

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(materialCellLabels.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());

	UT_VoxelTileIteratorI vitt;

	if (boss->opInterrupt())
	    return;

	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant() ||
		    vit.getValue() == HDK::Utilities::LIQUID_CELL)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			if (vitt.getValue() == HDK::Utilities::LIQUID_CELL)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			    fpreal divergence = 0;

			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I face = cellToFaceMap(cell, axis, direction);

				    fpreal sign = (direction % 2 == 0) ? 1. : -1.;
				    fpreal weight = getFieldValue(cutCellWeights[axis], face);

				    if (weight > 0)
				    {
					divergence += sign * weight * getFieldValue(*velocity.getField(axis), face);
					assert(getFieldValue(*validFaces.getField(axis), face) == HDK::Utilities::VALID_FACE);
				    }
				    if (solidVelocity != nullptr && weight < 1)
				    {
					UT_Vector3 point;
					velocity.getField(axis)->indexToPos(face[0], face[1], face[2], point);

					divergence += sign * (1. - weight) * solidVelocity->getField(axis)->getValue(point);
				    }
				}

			    debugDivergenceGrid.setValue(cell, divergence);
			}
		    }
		}
	    }
	}
    });  
}

void
HDK_FreeSurfacePressureSolver::buildMaxAndSumGrid(UT_Array<SolveReal> &tiledMaxDivergenceList,
						    UT_Array<SolveReal> &tiledSumDivergenceList,
						    const UT_VoxelArray<SolveReal> &debugDivergenceGrid,
						    const SIM_RawIndexField &materialCellLabels) const
{
    // Pre-expand all tiles that correspond to INTERIOR or BOUNDARY domain labels
    assert(debugDivergenceGrid.getVoxelRes() == materialCellLabels.getVoxelRes());

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(materialCellLabels.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());

	UT_VoxelTileIteratorI vitt;

	if (boss->opInterrupt())
	    return;

	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    SolveReal localMaxValue = 0;
	    SolveReal localSumValue = 0;

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant() ||
		    vit.getValue() == HDK::Utilities::LIQUID_CELL)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			if (vitt.getValue() == HDK::Utilities::LIQUID_CELL)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			    SolveReal value = debugDivergenceGrid(cell);

			    if (fabs(value) > fabs(localMaxValue))
				localMaxValue = value;

			    localSumValue += value;
			}
		    }

		    tiledMaxDivergenceList[tileNumber] = localMaxValue;
		    tiledSumDivergenceList[tileNumber] = localSumValue;
		}
	    }
	}
    });
}
