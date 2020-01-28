#include "HDK_GeometricFreeSurfacePressureSolver.h"

#include <PRM/PRM_Include.h>

#include <SIM/SIM_DopDescription.h>
#include <SIM/SIM_FieldUtils.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_PRMShared.h>

#include <UT/UT_DSOVersion.h>
#include <UT/UT_PerfMonAutoEvent.h>
#include <UT/UT_ThreadedAlgorithm.h>

#include "HDK_GeometricCGPoissonSolver.h"
#include "HDK_GeometricMultigridPoissonSolver.h"
#include "HDK_GeometricMultigridOperators.h"

#include "HDK_Utilities.h"

void
initializeSIM(void *)
{
   IMPLEMENT_DATAFACTORY(HDK_GeometricFreeSurfacePressureSolver);
}

// Standard constructor, note that BaseClass was crated by the
// DECLARE_DATAFACTORY and provides an easy way to chain through
// the class hierarchy.
HDK_GeometricFreeSurfacePressureSolver::HDK_GeometricFreeSurfacePressureSolver(const SIM_DataFactory *factory)
    : BaseClass(factory)
{
}

HDK_GeometricFreeSurfacePressureSolver::~HDK_GeometricFreeSurfacePressureSolver()
{
}

const SIM_DopDescription*
HDK_GeometricFreeSurfacePressureSolver::getDopDescription()
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

    static PRM_Name	theUseMGPreconditionerName("useMGPreconditioner", "Use Multigrid Preconditioner");

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

	PRM_Template(PRM_TOGGLE, 1, &theUseMGPreconditionerName, PRMoneDefaults),

    	PRM_Template()
    };

    static SIM_DopDescription theDopDescription(true,
						"HDK_GeometricFreeSurfacePressureSolver",
						"HDK Geometric Free Surface Pressure Solver",
						"$OS",
						classname(),
						theTemplates);

    setGasDescription(theDopDescription);

    return &theDopDescription;
}

bool
HDK_GeometricFreeSurfacePressureSolver::solveGasSubclass(SIM_Engine &engine,
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

    SIM_RawIndexField materialCellLabels;

    {
    	UT_PerfMonAutoSolveEvent event(this, "Build liquid cell labels");

    	std::cout << "\n// Build liquid cell labels" << std::endl;

    	materialCellLabels.match(liquidSurface);
	materialCellLabels.makeConstant(HDK::Utilities::UNLABELLED_CELL);

	HDK::Utilities::buildLiquidCellLabels(materialCellLabels,
						liquidSurface,
						*solidSurface,
						cutCellWeights);
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
    // Build domain labels with padding to use directly with an MG preconditioner
    //
    ////////////////////////////////////////////

    UT_VoxelArray<int> domainCellLabels;
    UT_Vector3I exteriorOffset;
    int mgLevels;
    {
    	UT_PerfMonAutoSolveEvent event(this, "Build MG domain labels");

    	std::cout << "\n// Build MG domain labels" << std::endl;

	std::pair<UT_Vector3I, int> mgSettings = buildMGDomainLabels(domainCellLabels, materialCellLabels);

	exteriorOffset = mgSettings.first;
	mgLevels = mgSettings.second;
    }

    ////////////////////////////////////////////
    //
    // Build right-hand-side for liquid degrees of freedom using the 
    // cut-cell method to account for moving solids.
    //
    ////////////////////////////////////////////

    UT_VoxelArray<StoreReal> rhsGrid;
    rhsGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
    rhsGrid.constant(0);

    {
    	std::cout << "\n// Build liquid RHS" << std::endl;
    	UT_PerfMonAutoSolveEvent event(this, "Build liquid right-hand side");

    	buildRHS(rhsGrid,
		    materialCellLabels,
		    domainCellLabels,
		    *velocity,
		    solidVelocity,
		    *validFaces,
		    cutCellWeights,
		    exteriorOffset);
    }

    ////////////////////////////////////////////
    //
    // Build solution grid and apply warm start
    //
    ////////////////////////////////////////////
    UT_VoxelArray<StoreReal> solutionGrid;
    solutionGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
    solutionGrid.constant(0);

    if (getUseOldPressure())
    {
	std::cout << "\n// Apply warm start pressure" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Apply warm start pressure");

	applyOldPressure(solutionGrid,
			    *pressure,
			    materialCellLabels,
			    domainCellLabels,
			    exteriorOffset);
    }

    ////////////////////////////////////////////
    //
    // Build weights for boundary conditions
    //
    ////////////////////////////////////////////

    std::array<UT_VoxelArray<StoreReal>, 3> boundaryWeights;

    {
	std::cout << "\n// Build boundary weights" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Build boundary weights");

	// We want to build expanded boundary weights that align with the MG domain
	for (int axis : {0,1,2})
	{
	    UT_Vector3I weightSize(domainCellLabels.getVoxelRes());
	    ++weightSize[axis];
	    boundaryWeights[axis].size(weightSize[0], weightSize[1], weightSize[2]);
	    boundaryWeights[axis].constant(0);

	    buildMGBoundaryWeights(boundaryWeights[axis],
				    cutCellWeights[axis],
				    *validFaces->getField(axis),
				    liquidSurface,
				    materialCellLabels,
				    domainCellLabels,
				    exteriorOffset,
				    axis);
	}
    }

    ////////////////////////////////////////////
    //
    // Set boundary domain labels
    //
    ////////////////////////////////////////////    
    
    {
	std::cout << "\n// Set boundary domain labels" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Set boundary domain labels");
	
	HDK::GeometricMultigridOperators::setBoundaryCellLabels(domainCellLabels, boundaryWeights);

	assert(HDK::GeometricMultigridOperators::unitTestBoundaryCells<StoreReal>(domainCellLabels, &boundaryWeights));
	assert(HDK::GeometricMultigridOperators::unitTestExteriorCells(domainCellLabels));
    }

    ////////////////////////////////////////////
    //
    // Solve for pressure
    //
    ////////////////////////////////////////////

    {
	std::cout << "\n// Solve liquid system" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Solve linear system");

	auto applyMatrixVectorMultiply = [&domainCellLabels, &boundaryWeights](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	{
	    return HDK::GeometricMultigridOperators::applyPoissonMatrix<SolveReal>(destination, source, domainCellLabels, &boundaryWeights);
	};

	auto applyDotProduct = [&domainCellLabels](const UT_VoxelArray<StoreReal> &grid0, const UT_VoxelArray<StoreReal> &grid1)
	{
	    assert(grid0.getVoxelRes() == grid1.getVoxelRes());
	    return HDK::GeometricMultigridOperators::dotProduct<SolveReal>(grid0, grid1, domainCellLabels);
	};

	auto getSquaredL2Norm = [&domainCellLabels](const UT_VoxelArray<StoreReal> &grid)
	{
	    return HDK::GeometricMultigridOperators::squaredL2Norm<SolveReal>(grid, domainCellLabels);
	};

	auto addToVector = [&domainCellLabels](UT_VoxelArray<StoreReal> &destination,
						const UT_VoxelArray<StoreReal> &source,
						const fpreal32 scale)
	{
	    HDK::GeometricMultigridOperators::addToVector<SolveReal>(destination, source, scale, domainCellLabels);
	};

	auto addScaledVector = [&domainCellLabels](UT_VoxelArray<StoreReal> &destination,
						    const UT_VoxelArray<StoreReal> &unscaledSource,
						    const UT_VoxelArray<StoreReal> &scaledSource,
						    const fpreal32 scale)
	{
	    HDK::GeometricMultigridOperators::addVectors<SolveReal>(destination, unscaledSource, scaledSource, scale, domainCellLabels);
	};

	if (getUseMGPreconditioner())
	{
	    HDK::GeometricMultigridPoissonSolver mgPreconditioner(domainCellLabels,
								    boundaryWeights,
								    mgLevels,
								    true /*use Gauss Seidel*/);

	    auto applyMultigridPreconditioner = [&mgPreconditioner](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	    {
		assert(destination.getVoxelRes() == source.getVoxelRes());
		mgPreconditioner.applyVCycle(destination, source);
	    };

	    HDK::solveGeometricConjugateGradient(solutionGrid,
						    rhsGrid,
						    applyMatrixVectorMultiply,
						    applyMultigridPreconditioner,
						    applyDotProduct,
						    getSquaredL2Norm,
						    addToVector,
						    addScaledVector,
						    SolveReal(getSolverTolerance()),
						    getMaxSolverIterations());
	}
	else
	{
	    UT_VoxelArray<StoreReal> diagonalPrecondGrid;
	    diagonalPrecondGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    diagonalPrecondGrid.constant(0);

	    {
		using HDK::GeometricMultigridOperators::CellLabels;
		using SIM::FieldUtils::cellToFaceMap;

		UT_Interrupt *boss = UTgetInterrupt();

		UTparallelForEachNumber(domainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
		{
		    UT_VoxelArrayIterator<int> vit;
		    vit.setConstArray(&domainCellLabels);

		    for (int i = range.begin(); i != range.end(); ++i)
		    {
			vit.myTileStart = i;
			vit.myTileEnd = i + 1;
			vit.rewind();

			if (boss->opInterrupt())
			    break;

			if (!vit.atEnd())
			{
			    if (!vit.isTileConstant() ||
				vit.getValue() == CellLabels::INTERIOR_CELL ||
				vit.getValue() == CellLabels::BOUNDARY_CELL)
			    {
				for (; !vit.atEnd(); vit.advance())
				{
				    if (vit.getValue() == CellLabels::INTERIOR_CELL)
				    {
					UT_Vector3I cell(vit.x(), vit.y(), vit.z());
#if !defined(NDEBUG)
					for (int axis : {0,1,2})
					    for (int direction : {0,1})
					    {
						UT_Vector3I face = cellToFaceMap(cell, axis, direction);
						assert(boundaryWeights[axis](face) == 1);

					    }
#endif

					diagonalPrecondGrid.setValue(cell, 1. / 6.);

				    }
				    else if (vit.getValue() == CellLabels::BOUNDARY_CELL)
				    {
					UT_Vector3I cell(vit.x(), vit.y(), vit.z());

					fpreal diagonal = 0;
					for (int axis : {0,1,2})
					    for (int direction : {0,1})
					    {
						UT_Vector3I face = cellToFaceMap(cell, axis, direction);
						diagonal += boundaryWeights[axis](face);
					    }
					diagonalPrecondGrid.setValue(cell, 1. / diagonal);
				    }
				}
			    }
			}
		    }
		});
	    }

	    auto applyDiagonalPreconditioner = [&domainCellLabels, &diagonalPrecondGrid](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	    {
		using HDK::GeometricMultigridOperators::CellLabels;

		assert(destination.getVoxelRes() == source.getVoxelRes() &&
			source.getVoxelRes() == domainCellLabels.getVoxelRes());

		UT_Interrupt *boss = UTgetInterrupt();

		UTparallelForEachNumber(domainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
		{
		    UT_VoxelArrayIterator<int> vit;
		    vit.setConstArray(&domainCellLabels);

		    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> destinationProbe;
		    destinationProbe.setArray(&destination);

		    UT_VoxelProbe<StoreReal, true /* read */, false /* write */, false> sourceProbe;
		    sourceProbe.setConstArray(&source);

		    UT_VoxelProbe<StoreReal, true /* read */, false /* write */, false> precondProbe;
		    precondProbe.setConstArray(&diagonalPrecondGrid);

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
				if (vit.getValue() == CellLabels::INTERIOR_CELL ||
				    vit.getValue() == CellLabels::BOUNDARY_CELL)
				{
				    destinationProbe.setIndex(vit);
				    sourceProbe.setIndex(vit);
				    precondProbe.setIndex(vit);

				    destinationProbe.setValue(fpreal(sourceProbe.getValue()) * fpreal(precondProbe.getValue()));
				}
			    }
			}
		    }
		});
	    };

	    HDK::solveGeometricConjugateGradient(solutionGrid,
						    rhsGrid,
						    applyMatrixVectorMultiply,
						    applyDiagonalPreconditioner,
						    applyDotProduct,
						    getSquaredL2Norm,
						    addToVector,
						    addScaledVector,
						    SolveReal(getSolverTolerance()),
						    getMaxSolverIterations());
	}
	
	UT_VoxelArray<StoreReal> residualGrid;
	residualGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	residualGrid.constant(0);

	// Compute r = b - Ax
	HDK::GeometricMultigridOperators::computePoissonResidual<SolveReal>(residualGrid, solutionGrid, rhsGrid, domainCellLabels, &boundaryWeights);

	std::cout << "  L-infinity error: " << HDK::GeometricMultigridOperators::infNorm(residualGrid, domainCellLabels) << std::endl;
	std::cout << "  Relative L-2: " << HDK::GeometricMultigridOperators::l2Norm<SolveReal>(residualGrid, domainCellLabels) / HDK::GeometricMultigridOperators::l2Norm<SolveReal>(rhsGrid, domainCellLabels) << std::endl;
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
	applySolutionToPressure(*pressure,
				solutionGrid,
				materialCellLabels,
				domainCellLabels,
				exteriorOffset);
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
				    materialCellLabels,
				    axis);
	}
    }

    {
	// Check divergence
	UT_VoxelArray<StoreReal> debugDivergenceGrid;
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
HDK_GeometricFreeSurfacePressureSolver::buildValidFaces(SIM_VectorField &validFaces,
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

std::pair<UT_Vector3I, int>
HDK_GeometricFreeSurfacePressureSolver::buildMGDomainLabels(UT_VoxelArray<int> &domainCellLabels,
							    const SIM_RawIndexField &materialCellLabels) const
{
    using HDK::GeometricMultigridOperators::CellLabels;

    // Build domain labels with the appropriate padding to apply
    // geometric multigrid directly without a wasteful transfer
    // for each v-cycle.

    // Cap MG levels at 4 voxels in the smallest dimension
    fpreal minLog = std::min(std::log2(fpreal(materialCellLabels.getVoxelRes()[0])),
				std::log2(fpreal(materialCellLabels.getVoxelRes()[1])));
    minLog = std::min(minLog, std::log2(fpreal(materialCellLabels.getVoxelRes()[2])));

    int mgLevels = ceil(minLog) - std::log2(fpreal(2));

    std::cout << "    MG levels: " << mgLevels << std::endl;

    // Add the necessary exterior cells so that after coarsening to the top level
    // there is still a single layer of exterior cells
    int exteriorPadding = std::pow(2, mgLevels - 1);

    UT_Vector3I expandedResolution = materialCellLabels.getVoxelRes() + 2 * UT_Vector3I(exteriorPadding);

    // Expand the domain to be a power of 2.
    for (int axis : {0,1,2})
    {
	fpreal logSize = std::log2(fpreal(expandedResolution[axis]));
	logSize = std::ceil(logSize);

	expandedResolution[axis] = exint(std::exp2(logSize));
    }
    
    UT_Vector3I exteriorOffset = UT_Vector3I(exteriorPadding);

    domainCellLabels.size(expandedResolution[0], expandedResolution[1], expandedResolution[2]);
    domainCellLabels.constant(CellLabels::EXTERIOR_CELL);

    // Build domain cell labels
    UT_Interrupt *boss = UTgetInterrupt();

    // Uncompress internal domain label tiles
    UT_Array<bool> isTileOccupiedList;
    isTileOccupiedList.setSize(domainCellLabels.numTiles());
    isTileOccupiedList.constant(false);

    UTparallelForEachNumber(materialCellLabels.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());
	UT_VoxelTileIteratorI vitt;

	for (int i = range.begin(); i != range.end(); ++i)
	{
	    vit.myTileStart = i;
	    vit.myTileEnd = i + 1;
	    vit.rewind();

	    if (boss->opInterrupt())
		break;

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant() ||
		    vit.getValue() != HDK::Utilities::UNLABELLED_CELL)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			if (vitt.getValue() != HDK::Utilities::UNLABELLED_CELL)
			{
			    UT_Vector3I cell = UT_Vector3I(vitt.x(), vitt.y(), vitt.z()) + exteriorOffset;

			    int tileNumber = domainCellLabels.indexToLinearTile(cell[0], cell[1], cell[2]);
			    if (!isTileOccupiedList[tileNumber])
				isTileOccupiedList[tileNumber] = true;
			}
		    }
		}
	    }
	}
    });

    HDK::GeometricMultigridOperators::uncompressTiles(domainCellLabels, isTileOccupiedList);

    // Copy initial domain labels to interior domain labels with padding
    UTparallelForEachNumber(materialCellLabels.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());

	UT_VoxelProbe<int, false /* no read */, true /* write */, true /* test for write */> localDomainProbe;
	localDomainProbe.setArray(&domainCellLabels);

	for (int i = range.begin(); i != range.end(); ++i)
	{
	    vit.myTileStart = i;
	    vit.myTileEnd = i + 1;
	    vit.rewind();

	    if (boss->opInterrupt())
		break;

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant() ||
		    vit.getValue() != HDK::Utilities::UNLABELLED_CELL)
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			const exint material = vit.getValue();
			if (material != HDK::Utilities::UNLABELLED_CELL)
			{
			    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
			    UT_Vector3I expandedCell = cell + exteriorOffset;

			    assert(!domainCellLabels.getLinearTile(domainCellLabels.indexToLinearTile(expandedCell[0], expandedCell[1], expandedCell[2]))->isConstant());

			    localDomainProbe.setIndex(expandedCell[0], expandedCell[1], expandedCell[2]);

			    if (material == HDK::Utilities::LIQUID_CELL)
				localDomainProbe.setValue(CellLabels::INTERIOR_CELL);
			    else
				localDomainProbe.setValue(CellLabels::DIRICHLET_CELL);
			}
		    }
		}
	    }
	}
    });

    domainCellLabels.collapseAllTiles();

    return std::pair<UT_Vector3I, int>(exteriorOffset, mgLevels);
}

void
HDK_GeometricFreeSurfacePressureSolver::buildRHS(UT_VoxelArray<StoreReal> &rhsGrid,
						    const SIM_RawIndexField &materialCellLabels,
						    const UT_VoxelArray<int> &domainCellLabels,
						    const SIM_VectorField &velocity,
						    const SIM_VectorField *solidVelocity,
						    const SIM_VectorField &validFaces,
						    const std::array<SIM_RawField, 3> &cutCellWeights,
						    const UT_Vector3I exteriorOffset) const
{
    using SIM::FieldUtils::cellToFaceMap;
    using SIM::FieldUtils::getFieldValue;
    using HDK::GeometricMultigridOperators::CellLabels;

    // Pre-expand all tiles that correspond to INTERIOR or BOUNDARY domain labels
    assert(rhsGrid.getVoxelRes() == domainCellLabels.getVoxelRes());
    HDK::GeometricMultigridOperators::uncompressActiveGrid(rhsGrid, domainCellLabels);

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
			    UT_Vector3I expandedCell = cell + exteriorOffset;
			    assert(!rhsGrid.getLinearTile(rhsGrid.indexToLinearTile(expandedCell[0],
										    expandedCell[1],
										    expandedCell[2]))->isConstant());
			    assert(domainCellLabels(expandedCell) == CellLabels::INTERIOR_CELL);

			    rhsGrid.setValue(expandedCell, divergence);
			}
		    }
		}
	    }
	}
    });  
}

void
HDK_GeometricFreeSurfacePressureSolver::applyOldPressure(UT_VoxelArray<StoreReal> &solutionGrid,
							    const SIM_RawField &pressure,
							    const SIM_RawIndexField &materialCellLabels,
							    const UT_VoxelArray<int> &domainCellLabels,
							    const UT_Vector3I &exteriorOffset) const
{
    using SIM::FieldUtils::getFieldValue;
    using HDK::GeometricMultigridOperators::CellLabels;

    // Pre-expand all tiles that correspond to INTERIOR or BOUNDARY domain labels
    assert(solutionGrid.getVoxelRes() == domainCellLabels.getVoxelRes());
    HDK::GeometricMultigridOperators::uncompressActiveGrid(solutionGrid, domainCellLabels);

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
		if (!vit.isTileConstant() || vit.getValue() == HDK::Utilities::LIQUID_CELL)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			if (vitt.getValue() == HDK::Utilities::LIQUID_CELL)
			{
			    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

			    UT_Vector3I expandedCell = cell + exteriorOffset;
			    assert(!solutionGrid.getLinearTile(solutionGrid.indexToLinearTile(expandedCell[0],
												expandedCell[1],
												expandedCell[2]))->isConstant());
			    assert(domainCellLabels(expandedCell) == CellLabels::INTERIOR_CELL);
			    solutionGrid.setValue(expandedCell, getFieldValue(pressure, cell));
			}
		    }
		}
	    }
	}
    });
}

void
HDK_GeometricFreeSurfacePressureSolver::buildMGBoundaryWeights(UT_VoxelArray<StoreReal> &boundaryWeights,
								const SIM_RawField &cutCellWeights,
								const SIM_RawField &validFaces,
								const SIM_RawField &liquidSurface,
								const SIM_RawIndexField &materialCellLabels,
								const UT_VoxelArray<int> &domainCellLabels,
								const UT_Vector3I &exteriorOffset,
								const int axis) const
{
    using SIM::FieldUtils::setFieldValue;
    using SIM::FieldUtils::getFieldValue;
    using SIM::FieldUtils::faceToCellMap;
    using HDK::GeometricMultigridOperators::CellLabels;

    UT_Interrupt *boss = UTgetInterrupt();

    UT_Vector3I voxelRes = materialCellLabels.getVoxelRes();

    UT_Array<bool> isTileOccupiedList;
    isTileOccupiedList.setSize(boundaryWeights.numTiles());
    isTileOccupiedList.constant(false);

    UTparallelForEachNumber(validFaces.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorF vit;
	vit.setConstArray(validFaces.field());

	UT_VoxelTileIteratorF vitt;

	if (boss->opInterrupt())
	    return;
	 
	for (int i = range.begin(); i != range.end(); ++i)
	{
	    vit.myTileStart = i;
	    vit.myTileEnd = i + 1;
	    vit.rewind();

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant() ||
		    vit.getValue() == HDK::Utilities::VALID_FACE)
		{
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {	      
			if (vitt.getValue() == HDK::Utilities::VALID_FACE)
			{
			    UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());

			    UT_Vector3I expandedFace = face + exteriorOffset;

			    int tileNumber = boundaryWeights.indexToLinearTile(expandedFace[0], expandedFace[1], expandedFace[2]);

			    if (!isTileOccupiedList[tileNumber])
				isTileOccupiedList[tileNumber] = true;
			}
		    }
		}
	    }
	}
    });
    
    HDK::GeometricMultigridOperators::uncompressTiles(boundaryWeights, isTileOccupiedList);

    UTparallelForEachNumber(validFaces.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorF vit;
	vit.setConstArray(validFaces.field());

	UT_VoxelTileIteratorF vitt;

	if (boss->opInterrupt())
	    return;
	 
	for (int i = range.begin(); i != range.end(); ++i)
	{
	    vit.myTileStart = i;
	    vit.myTileEnd = i + 1;
	    vit.rewind();

	    if (!vit.atEnd())
	    {
		if (!vit.isTileConstant() ||
		    vit.getValue() == HDK::Utilities::VALID_FACE)
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
				if (backwardCell[axis] < 0)
				{
				    assert(domainCellLabels(backwardCell + exteriorOffset) == CellLabels::EXTERIOR_CELL);
				    assert(domainCellLabels(forwardCell + exteriorOffset) == CellLabels::INTERIOR_CELL);
				}
				else
				{
				    assert(domainCellLabels(backwardCell + exteriorOffset) == CellLabels::INTERIOR_CELL);
				    assert(domainCellLabels(forwardCell + exteriorOffset) == CellLabels::EXTERIOR_CELL);
				}

				boundaryWeights.setValue(face + exteriorOffset, 0);
				continue;
			    }

			    exint backwardMaterial = getFieldValue(materialCellLabels, backwardCell);
			    exint forwardMaterial = getFieldValue(materialCellLabels, forwardCell);

			    assert(backwardMaterial == HDK::Utilities::LIQUID_CELL ||
				    forwardMaterial == HDK::Utilities::LIQUID_CELL);

			    fpreal weight = getFieldValue(cutCellWeights, face);
			    assert(weight > 0);

			    if (backwardMaterial != HDK::Utilities::LIQUID_CELL ||
				forwardMaterial != HDK::Utilities::LIQUID_CELL)
			    {
				if (backwardMaterial != HDK::Utilities::LIQUID_CELL)
				{
				    assert(domainCellLabels(backwardCell + exteriorOffset) != CellLabels::INTERIOR_CELL);
				    assert(domainCellLabels(forwardCell + exteriorOffset) == CellLabels::INTERIOR_CELL);
				}
				else
				{
				    assert(domainCellLabels(backwardCell + exteriorOffset) == CellLabels::INTERIOR_CELL);
				    assert(domainCellLabels(forwardCell + exteriorOffset) != CellLabels::INTERIOR_CELL);
				}

				fpreal backwardPhi = getFieldValue(liquidSurface, backwardCell);
				fpreal forwardPhi = getFieldValue(liquidSurface, forwardCell);

				fpreal theta = HDK::Utilities::computeGhostFluidWeight(backwardPhi, forwardPhi);
				theta = SYSclamp(theta, .01, 1.);
				weight /= theta;
			    }
			    else
			    {
				assert(domainCellLabels(backwardCell + exteriorOffset) == CellLabels::INTERIOR_CELL &&
					domainCellLabels(forwardCell + exteriorOffset) == CellLabels::INTERIOR_CELL);
			    }

			    UT_Vector3I expandedFace = face + exteriorOffset;
			    assert(!boundaryWeights.getLinearTile(boundaryWeights.indexToLinearTile(expandedFace[0],
												    expandedFace[1],
												    expandedFace[2]))->isConstant());
			    boundaryWeights.setValue(expandedFace, weight);
			}
		    }
		}
	    }
	}
    });
}

void
HDK_GeometricFreeSurfacePressureSolver::applySolutionToPressure(SIM_RawField &pressure,
							const UT_VoxelArray<StoreReal> &solutionGrid,
							const SIM_RawIndexField &materialCellLabels,
							const UT_VoxelArray<int> &domainCellLabels,
							const UT_Vector3I &exteriorOffset) const
{
    using SIM::FieldUtils::setFieldValue;
    using HDK::GeometricMultigridOperators::CellLabels;

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
			    UT_Vector3I expandedCell = cell + exteriorOffset;

			    assert(domainCellLabels(expandedCell) == CellLabels::INTERIOR_CELL ||
				    domainCellLabels(expandedCell) == CellLabels::BOUNDARY_CELL);

			    setFieldValue(pressure, cell, solutionGrid(expandedCell));
			}
		    }
		}
	    }
	}
    });
}

void
HDK_GeometricFreeSurfacePressureSolver::applyPressureGradient(SIM_RawField &velocity,
								const SIM_RawField &validFaces,
								const SIM_RawField &pressure,
								const SIM_RawField &liquidSurface,
								const SIM_RawField &cutCellWeights,
								const SIM_RawIndexField &materialCellLabels,
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

			    exint backwardMaterial = getFieldValue(materialCellLabels, backwardCell);
			    exint forwardMaterial = getFieldValue(materialCellLabels, forwardCell);

			    assert(backwardMaterial == HDK::Utilities::LIQUID_CELL || forwardMaterial >= HDK::Utilities::LIQUID_CELL);
			    assert(getFieldValue(cutCellWeights, face) > 0);

			    fpreal gradient = getFieldValue(pressure, forwardCell) - getFieldValue(pressure, backwardCell);

			    if (backwardMaterial != HDK::Utilities::LIQUID_CELL ||
				forwardMaterial != HDK::Utilities::LIQUID_CELL)
			    {
				if (backwardMaterial == HDK::Utilities::LIQUID_CELL)
				    assert(forwardMaterial == HDK::Utilities::AIR_CELL);
				else
				{
				    assert(backwardMaterial == HDK::Utilities::AIR_CELL);
				    assert(forwardMaterial == HDK::Utilities::LIQUID_CELL);
				}

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
HDK_GeometricFreeSurfacePressureSolver::buildDebugDivergence(UT_VoxelArray<StoreReal> &debugDivergenceGrid,
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
HDK_GeometricFreeSurfacePressureSolver::buildMaxAndSumGrid(UT_Array<SolveReal> &tiledMaxDivergenceList,
							    UT_Array<SolveReal> &tiledSumDivergenceList,
							    const UT_VoxelArray<StoreReal> &debugDivergenceGrid,
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
