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
    
    std::array<const SIM_RawField *, 3> cutCellWeights;
    for (int axis : {0, 1, 2})
	cutCellWeights[axis] = cutCellWeightsField->getField(axis);

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

	HDK::Utilities::buildMaterialCellLabels(materialCellLabels,
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
		    cutCellWeights);
    }

    // TODO: project out null space if there are no free surface cels.
    // This is important for smoke simulations.

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
			    liquidSurface,
			    liquidCellIndices,
			    materialCellLabels,			    
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

	Eigen::SparseMatrix<SolveReal> poissonMatrix(liquidCellCount, liquidCellCount);
	poissonMatrix.setFromTriplets(poissonElements.begin(), poissonElements.end());
	poissonMatrix.makeCompressed();
	
	Eigen::ConjugateGradient<Eigen::SparseMatrix<SolveReal>, Eigen::Lower | Eigen::Upper> solver;
	solver.compute(poissonMatrix);
    	solver.setTolerance(getSolverTolerance());

	solutionVector = solver.solveWithGuess(rhsVector, solutionVector);
	
	Vector residualVector = rhsVector - poissonMatrix * solutionVector;

	std::cout << "Solver iterations: " << solver.iterations() << std::endl;
	std::cout << "Solver error: " << solver.error() << std::endl;
	std::cout << "Recomputed relative residual: " << std::sqrt(residualVector.squaredNorm() / rhsVector.squaredNorm()) << std::endl;
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
    // Apply pressure gradient to velocity field
    //
    ////////////////////////////////////////////

    {
	std::cout << "\n// Update velocity from pressure gradient" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Update velocity from pressure gradient");

	for (int axis : {0,1,2})
	{
	    applyPressureGradient(*velocity->getField(axis),
				    *cutCellWeights[axis],
				    liquidSurface,
				    *pressure,
				    *validFaces->getField(axis),
				    liquidCellIndices,
				    axis);
	}
    }

    {
	UT_PerfMonAutoSolveEvent event(this, "Verify divergence-free constraint");
	std::cout << "\n// Verify divergence-free constraint" << std::endl;

	UT_Array<SolveReal> parallelAccumulatedDivergence;
	parallelAccumulatedDivergence.setSize(threadCount);
	parallelAccumulatedDivergence.constant(0);

	UT_Array<SolveReal> parallelMaxDivergence;
	parallelMaxDivergence.setSize(threadCount);
	parallelMaxDivergence.constant(0);

	UT_Array<SolveReal> parallelCellCount;
	parallelCellCount.setSize(threadCount);
	parallelCellCount.constant(0);

	computeResultingDivergence(parallelAccumulatedDivergence,
				    parallelCellCount,
				    parallelMaxDivergence,
				    materialCellLabels,
				    *velocity,
				    solidVelocity,
				    cutCellWeights);

	SolveReal accumulatedDivergence = 0;
	SolveReal maxDivergence = 0;
	SolveReal cellCount = 0;

	for (int thread = 0; thread < threadCount; ++thread)
	{
	    accumulatedDivergence += parallelAccumulatedDivergence[thread];
	    maxDivergence = std::max(maxDivergence, parallelMaxDivergence[thread]);
	    cellCount += parallelCellCount[thread];
	}

	std::cout << "    Max divergence: " << maxDivergence << std::endl;
	std::cout << "    Accumulated divergence: " << accumulatedDivergence << std::endl;
	std::cout << "    Average divergence: " << accumulatedDivergence / cellCount << std::endl;
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
					const std::array<const SIM_RawField *, 3> &cutCellWeights) const
{
    using SIM::FieldUtils::cellToFaceMap;
    using SIM::FieldUtils::getFieldValue;

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(liquidCellIndices.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(liquidCellIndices.field());

	if (boss->opInterrupt())
	    return;

	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();
		
	    if (!vit.isTileConstant())
	    {
		for (; !vit.atEnd(); vit.advance())
		{
		    exint liquidIndex = vit.getValue();
		    if (liquidIndex >= 0)
		    {
			UT_Vector3I cell(vit.x(), vit.y(), vit.z());

			assert(getFieldValue(materialCellLabels, cell) == MaterialLabels::LIQUID_CELL);

			SolveReal divergence = 0;

			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I face = cellToFaceMap(cell, axis, direction);

				SolveReal sign = (direction == 0) ? 1. : -1.;
				SolveReal weight = getFieldValue(*cutCellWeights[axis], face);

				if (weight > 0)
				    divergence += sign * weight * getFieldValue(*velocity.getField(axis), face);
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
    });  
}

void
HDK_FreeSurfacePressureSolver::buildPoissonRows(std::vector<std::vector<Eigen::Triplet<SolveReal>>> &parallelPoissonElements,
						const SIM_RawField &liquidSurface,
						const SIM_RawIndexField &liquidCellIndices,	
						const SIM_RawIndexField &materialCellLabels,						
						const std::array<const SIM_RawField *, 3> &cutCellWeights) const
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
			SolveReal poissonDiagonal = 0.;
			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				if (adjacentCell[axis] < 0 || adjacentCell[axis] >= voxelRes[axis])
				    continue;

				UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				SolveReal weight = getFieldValue(*cutCellWeights[axis], face);

				if (weight > 0)
				{
				    assert(weight > 0);
				
				    exint adjacentLiquidIndex = getFieldValue(liquidCellIndices, adjacentCell);

				    if (adjacentLiquidIndex >= 0)
				    {
					assert(getFieldValue(materialCellLabels, adjacentCell) == MaterialLabels::LIQUID_CELL);
					poissonDiagonal += weight;
					localPoissonElements.push_back(Eigen::Triplet<SolveReal>(liquidIndex, adjacentLiquidIndex, -weight));
				    }
				    else
				    {
					SolveReal phi0 = getFieldValue(liquidSurface, cell);
					SolveReal phi1 = getFieldValue(liquidSurface, adjacentCell);

					assert(phi1 > 0);
					assert(getFieldValue(materialCellLabels, adjacentCell) == MaterialLabels::AIR_CELL);

					SolveReal theta = HDK::Utilities::computeGhostFluidWeight(phi0, phi1);
					theta = SYSclamp(theta, .01, 1.);

					poissonDiagonal += weight / theta;
				    }
				}
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

	if (boss->opInterrupt())
	    return;
	 
	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.isTileConstant())
	    {
		for (; !vit.atEnd(); vit.advance())
		{
		    exint liquidIndex = vit.getValue();

		    if (liquidIndex >= 0)
		    {
			UT_Vector3I cell(vit.x(), vit.y(), vit.z());
			solutionVector(liquidIndex) = getFieldValue(pressure, cell);
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

	if (boss->opInterrupt())
	    return;
	 
	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.isTileConstant())
	    {
		for (; !vit.atEnd(); vit.advance())
		{
		    exint liquidIndex = vit.getValue();

		    if (liquidIndex >= 0)
		    {
			UT_Vector3I cell(vit.x(), vit.y(), vit.z());
			setFieldValue(pressure, cell, solutionVector(liquidIndex));
		    }
		}
	    }
	}
    });
}

void
HDK_FreeSurfacePressureSolver::buildValidFaces(SIM_VectorField &validFaces,
						const SIM_RawIndexField &materialCellLabels,
						const std::array<const SIM_RawField *, 3> &cutCellWeights) const
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
						[](const exint label){ return label == MaterialLabels::LIQUID_CELL; },
						axis);

	HDK::Utilities::uncompressTiles(*validFaces.getField(axis), isTileOccupiedList);

	HDK::Utilities::classifyValidFaces(*validFaces.getField(axis),
					    materialCellLabels,
					    *cutCellWeights[axis],
					    [](const exint label){ return label == MaterialLabels::LIQUID_CELL; },
					    axis);
    }
}

void
HDK_FreeSurfacePressureSolver::applyPressureGradient(SIM_RawField &velocity,
							const SIM_RawField &cutCellWeights,	
							const SIM_RawField &liquidSurface,	
							const SIM_RawField &pressure,
							const SIM_RawField &validFaces,					
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

	if (boss->opInterrupt())
	    return;
	 
	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.isTileConstant() || vit.getValue() == HDK::Utilities::VALID_FACE)
	    {
		for (; !vit.atEnd(); vit.advance())
		{	      
		    if (vit.getValue() == HDK::Utilities::VALID_FACE)
		    {
			UT_Vector3I face(vit.x(), vit.y(), vit.z());

			UT_Vector3I backwardCell = faceToCellMap(face, axis, 0);
			UT_Vector3I forwardCell = faceToCellMap(face, axis, 1);

			if (backwardCell[axis] < 0 || forwardCell[axis] >= voxelRes[axis])
			    continue;

			exint backwardIndex = getFieldValue(liquidCellIndices, backwardCell);
			exint forwardIndex = getFieldValue(liquidCellIndices, forwardCell);

			assert(backwardIndex >= 0 || forwardIndex >= 0);
			assert(getFieldValue(cutCellWeights, face) > 0);

			SolveReal gradient = getFieldValue(pressure, forwardCell) - getFieldValue(pressure, backwardCell);

			if (backwardIndex == HDK::Utilities::UNLABELLED_CELL ||
			    forwardIndex == HDK::Utilities::UNLABELLED_CELL)
			{
			    SolveReal backwardPhi = getFieldValue(liquidSurface, backwardCell);
			    SolveReal forwardPhi = getFieldValue(liquidSurface, forwardCell);

			    SolveReal theta = HDK::Utilities::computeGhostFluidWeight(backwardPhi, forwardPhi);
			    theta = SYSclamp(theta, .01, 1.);

			    gradient /= theta;
			}

			setFieldValue(velocity, face, getFieldValue(velocity, face) - gradient);
		    }
		}
	    }
	}
    });
}

void
HDK_FreeSurfacePressureSolver::computeResultingDivergence(UT_Array<SolveReal> &parallelAccumulatedDivergence,
							    UT_Array<SolveReal> &parallelCellCount,
							    UT_Array<SolveReal> &parallelMaxDivergence,							    
							    const SIM_RawIndexField &materialCellLabels,
							    const SIM_VectorField &velocity,
							    const SIM_VectorField *solidVelocity,
							    const std::array<const SIM_RawField *, 3> &cutCellWeights) const
{
    using SIM::FieldUtils::cellToCellMap;
    using SIM::FieldUtils::cellToFaceMap;
    using SIM::FieldUtils::getFieldValue;

    UT_Interrupt *boss = UTgetInterrupt();

    UT_ThreadedAlgorithm computeResultingDivergenceAlgorithm;
    computeResultingDivergenceAlgorithm.run([&](const UT_JobInfo &info)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());
	vit.splitByTile(info);

	UT_VoxelTileIteratorI vitt;

	SolveReal &localAccumulatedDivergence = parallelAccumulatedDivergence[info.job()];
	SolveReal &localMaxDivergence = parallelMaxDivergence[info.job()];
	SolveReal &localCellCount = parallelCellCount[info.job()];

	for (vit.rewind(); !vit.atEnd(); vit.advanceTile())
	{
	    if (boss->opInterrupt())
		return 0;

	    if (!vit.isTileConstant() ||
		vit.getValue() == MaterialLabels::LIQUID_CELL)
	    {
		vitt.setTile(vit);

		for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		{
		    if (vitt.getValue() == MaterialLabels::LIQUID_CELL)
		    {
			UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			SolveReal divergence = 0;

			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				SolveReal weight = getFieldValue(*cutCellWeights[axis], face);

				SolveReal sign = (direction == 0) ? -1 : 1;

				if (weight > 0)
				    divergence += sign * weight * getFieldValue(*velocity.getField(axis), face);
				if (solidVelocity != nullptr && weight < 1)
				{
				    UT_Vector3 point;
				    velocity.getField(axis)->indexToPos(face[0], face[1], face[2], point);

				    divergence += sign * (1. - weight) * solidVelocity->getField(axis)->getValue(point);
				}
			    }

			localAccumulatedDivergence += divergence;
			localMaxDivergence = std::max(localMaxDivergence, divergence);
			++localCellCount;
		    }
		}
	    }
	}

	return 0;
    });
}