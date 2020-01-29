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

    fpreal32 constantLiquidDensity;
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

	HDK::Utilities::buildMaterialCellLabels(materialCellLabels,
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

    UT_VoxelArray<int> mgDomainCellLabels;
    std::array<UT_VoxelArray<SolveReal>, 3> mgBoundaryWeights;

    UT_Vector3I mgExpandedOffset;
    int mgLevels;

    {
    	UT_PerfMonAutoSolveEvent event(this, "Build MG domain labels and weights");
	std::cout << "\n// Build MG domain labels and weights" << std::endl;

	using MGCellLabels = HDK::GeometricMultigridOperators::CellLabels;

	UT_VoxelArray<int> baseDomainCellLabels;
	baseDomainCellLabels.size(materialCellLabels.getVoxelRes()[0],
				    materialCellLabels.getVoxelRes()[1],
				    materialCellLabels.getVoxelRes()[2]);

	baseDomainCellLabels.constant(MGCellLabels::EXTERIOR_CELL);

	buildMGDomainLabels(baseDomainCellLabels,
			    materialCellLabels);

	// Build boundary weights
	std::array<UT_VoxelArray<SolveReal>, 3> baseBoundaryWeights;

	for (int axis : {0,1,2})
	{
	    UT_Vector3I size = baseDomainCellLabels.getVoxelRes();
	    ++size[axis];

	    baseBoundaryWeights[axis].size(size[0], size[1], size[2]);
	    baseBoundaryWeights[axis].constant(0);

	    buildMGBoundaryWeights(baseBoundaryWeights[axis],
				    *cutCellWeights[axis],
				    liquidSurface,
				    *validFaces->getField(axis),
				    materialCellLabels,
				    baseDomainCellLabels,
				    axis);
	}

	// Build expanded domain
	auto isExteriorCell = [](const int value) { return value == MGCellLabels::EXTERIOR_CELL; };
	auto isInteriorCell = [](const int value) { return value == MGCellLabels::INTERIOR_CELL; };
	auto isDirichletCell = [](const int value) { return value == MGCellLabels::DIRICHLET_CELL; };

	std::pair<UT_Vector3I, int> mgSettings = HDK::GeometricMultigridOperators::buildExpandedCellLabels(mgDomainCellLabels, baseDomainCellLabels, isExteriorCell, isInteriorCell, isDirichletCell);

	mgExpandedOffset = mgSettings.first;
	mgLevels = mgSettings.second;
	
	// Build expanded boundary weights
	for (int axis : {0,1,2})
	{
	    UT_Vector3I size = mgDomainCellLabels.getVoxelRes();
	    ++size[axis];

	    mgBoundaryWeights[axis].size(size[0], size[1], size[2]);
	    mgBoundaryWeights[axis].constant(0);

	    HDK::GeometricMultigridOperators::buildExpandedBoundaryWeights(mgBoundaryWeights[axis], baseBoundaryWeights[axis], mgDomainCellLabels, mgExpandedOffset, axis);
	}

	// Build boundary cells
	HDK::GeometricMultigridOperators::setBoundaryCellLabels(mgDomainCellLabels, mgBoundaryWeights);

	assert(HDK::GeometricMultigridOperators::unitTestBoundaryCells<SolveReal>(mgDomainCellLabels, &mgBoundaryWeights));
	assert(HDK::GeometricMultigridOperators::unitTestExteriorCells(mgDomainCellLabels));
    }

    ////////////////////////////////////////////
    //
    // Build right-hand-side for liquid degrees of freedom using the 
    // cut-cell method to account for moving solids.
    //
    ////////////////////////////////////////////

    UT_VoxelArray<StoreReal> rhsGrid;
    rhsGrid.size(mgDomainCellLabels.getVoxelRes()[0],
		    mgDomainCellLabels.getVoxelRes()[1],
		    mgDomainCellLabels.getVoxelRes()[2]);

    rhsGrid.constant(0);

    {
    	std::cout << "\n// Build liquid RHS" << std::endl;
    	UT_PerfMonAutoSolveEvent event(this, "Build liquid right-hand side");

    	buildRHS(rhsGrid,
		    materialCellLabels,
		    *velocity,
		    solidVelocity,
		    cutCellWeights,
		    mgDomainCellLabels,
		    mgExpandedOffset);
    }

    ////////////////////////////////////////////
    //
    // Build solution grid and apply warm start
    //
    ////////////////////////////////////////////

    UT_VoxelArray<StoreReal> solutionGrid;
    solutionGrid.size(mgDomainCellLabels.getVoxelRes()[0],
			mgDomainCellLabels.getVoxelRes()[1],
			mgDomainCellLabels.getVoxelRes()[2]);

    solutionGrid.constant(0);

    if (getUseOldPressure())
    {
	std::cout << "\n// Apply warm start pressure" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Apply warm start pressure");

	applyOldPressure(solutionGrid,
			    *pressure,
			    materialCellLabels,
			    mgDomainCellLabels,
			    mgExpandedOffset);
    }

    ////////////////////////////////////////////
    //
    // Solve for pressure
    //
    ////////////////////////////////////////////

    {
	std::cout << "\n// Solve liquid system" << std::endl;
	UT_PerfMonAutoSolveEvent event(this, "Solve linear system");

	auto applyMatrixVectorMultiply = [&mgDomainCellLabels, &mgBoundaryWeights](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	{
	    return HDK::GeometricMultigridOperators::applyPoissonMatrix<SolveReal>(destination, source, mgDomainCellLabels, &mgBoundaryWeights);
	};

	auto applyDotProduct = [&mgDomainCellLabels](const UT_VoxelArray<StoreReal> &grid0, const UT_VoxelArray<StoreReal> &grid1)
	{
	    assert(grid0.getVoxelRes() == grid1.getVoxelRes());
	    return HDK::GeometricMultigridOperators::dotProduct<SolveReal>(grid0, grid1, mgDomainCellLabels);
	};

	auto getSquaredL2Norm = [&mgDomainCellLabels](const UT_VoxelArray<StoreReal> &grid)
	{
	    return HDK::GeometricMultigridOperators::squaredL2Norm<SolveReal>(grid, mgDomainCellLabels);
	};

	auto addToVector = [&mgDomainCellLabels](UT_VoxelArray<StoreReal> &destination,
						const UT_VoxelArray<StoreReal> &source,
						const SolveReal scale)
	{
	    HDK::GeometricMultigridOperators::addToVector<SolveReal>(destination, source, scale, mgDomainCellLabels);
	};

	auto addScaledVector = [&mgDomainCellLabels](UT_VoxelArray<StoreReal> &destination,
						    const UT_VoxelArray<StoreReal> &unscaledSource,
						    const UT_VoxelArray<StoreReal> &scaledSource,
						    const SolveReal scale)
	{
	    HDK::GeometricMultigridOperators::addVectors<SolveReal>(destination, unscaledSource, scaledSource, scale, mgDomainCellLabels);
	};

	if (getUseMGPreconditioner())
	{
	    HDK::GeometricMultigridPoissonSolver mgPreconditioner(mgDomainCellLabels,
								    mgBoundaryWeights,
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
	    diagonalPrecondGrid.size(mgDomainCellLabels.getVoxelRes()[0],
					mgDomainCellLabels.getVoxelRes()[1],
					mgDomainCellLabels.getVoxelRes()[2]);

	    diagonalPrecondGrid.constant(0);

	    {
		using MGCellLabels = HDK::GeometricMultigridOperators::CellLabels;
		using SIM::FieldUtils::cellToFaceMap;

		UT_Interrupt *boss = UTgetInterrupt();

		UTparallelForEachNumber(mgDomainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
		{
		    UT_VoxelArrayIterator<int> vit;
		    vit.setConstArray(&mgDomainCellLabels);

		    for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
		    {
			vit.myTileStart = tileNumber;
			vit.myTileEnd = tileNumber + 1;
			vit.rewind();

			if (boss->opInterrupt())
			    break;

			if (!vit.isTileConstant() ||
			    vit.getValue() == MGCellLabels::INTERIOR_CELL ||
			    vit.getValue() == MGCellLabels::BOUNDARY_CELL)
			{
			    for (; !vit.atEnd(); vit.advance())
			    {
				if (vit.getValue() == MGCellLabels::INTERIOR_CELL)
				{
				    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
#if !defined(NDEBUG)
				    for (int axis : {0,1,2})
					for (int direction : {0,1})
					{
					    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
					    assert(mgBoundaryWeights[axis](face) == 1);

					}
#endif

				    diagonalPrecondGrid.setValue(cell, 1. / 6.);

				}
				else if (vit.getValue() == MGCellLabels::BOUNDARY_CELL)
				{
				    UT_Vector3I cell(vit.x(), vit.y(), vit.z());

				    SolveReal diagonal = 0;
				    for (int axis : {0,1,2})
					for (int direction : {0,1})
					{
					    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
					    diagonal += mgBoundaryWeights[axis](face);
					}
				    diagonalPrecondGrid.setValue(cell, 1. / diagonal);
				}
			    }
			}
		    }
		});
	    }

	    auto applyDiagonalPreconditioner = [&mgDomainCellLabels, &diagonalPrecondGrid](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	    {
		using MGCellLabels = HDK::GeometricMultigridOperators::CellLabels;

		assert(destination.getVoxelRes() == source.getVoxelRes() &&
			source.getVoxelRes() == mgDomainCellLabels.getVoxelRes());

		UT_Interrupt *boss = UTgetInterrupt();

		UTparallelForEachNumber(mgDomainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
		{
		    UT_VoxelArrayIterator<int> vit;
		    vit.setConstArray(&mgDomainCellLabels);

		    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> destinationProbe;
		    destinationProbe.setArray(&destination);

		    UT_VoxelProbe<StoreReal, true /* read */, false /* write */, false> sourceProbe;
		    sourceProbe.setConstArray(&source);

		    UT_VoxelProbe<StoreReal, true /* read */, false /* write */, false> precondProbe;
		    precondProbe.setConstArray(&diagonalPrecondGrid);

		    for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
		    {
			vit.myTileStart = tileNumber;
			vit.myTileEnd = tileNumber + 1;
			vit.rewind();

			if (boss->opInterrupt())
			    break;

			if (!vit.isTileConstant() ||
			    vit.getValue() == MGCellLabels::INTERIOR_CELL ||
			    vit.getValue() == MGCellLabels::BOUNDARY_CELL)
			{
			    for (; !vit.atEnd(); vit.advance())
			    {
				if (vit.getValue() == MGCellLabels::INTERIOR_CELL ||
				    vit.getValue() == MGCellLabels::BOUNDARY_CELL)
				{
				    destinationProbe.setIndex(vit);
				    sourceProbe.setIndex(vit);
				    precondProbe.setIndex(vit);

				    destinationProbe.setValue(SolveReal(sourceProbe.getValue()) * SolveReal(precondProbe.getValue()));
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
	residualGrid.size(mgDomainCellLabels.getVoxelRes()[0], mgDomainCellLabels.getVoxelRes()[1], mgDomainCellLabels.getVoxelRes()[2]);
	residualGrid.constant(0);

	// Compute r = b - Ax
	HDK::GeometricMultigridOperators::computePoissonResidual<SolveReal>(residualGrid, solutionGrid, rhsGrid, mgDomainCellLabels, &mgBoundaryWeights);

	std::cout << "  L-infinity error: " << HDK::GeometricMultigridOperators::infNorm(residualGrid, mgDomainCellLabels) << std::endl;
	std::cout << "  Relative L-2: " << HDK::GeometricMultigridOperators::l2Norm<SolveReal>(residualGrid, mgDomainCellLabels) / HDK::GeometricMultigridOperators::l2Norm<SolveReal>(rhsGrid, mgDomainCellLabels) << std::endl;
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
				materialCellLabels,
				mgDomainCellLabels,
				solutionGrid,
				mgExpandedOffset);
    }

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
				    materialCellLabels,
				    axis);
	}
    }

    {
	UT_PerfMonAutoSolveEvent event(this, "Verify divergence-free constraint");
	std::cout << "\n// Verify divergence-free constraint" << std::endl;

	const int threadCount = UT_Thread::getNumProcessors();

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
HDK_GeometricFreeSurfacePressureSolver::buildValidFaces(SIM_VectorField &validFaces,
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
HDK_GeometricFreeSurfacePressureSolver::buildMGDomainLabels(UT_VoxelArray<int> &mgDomainCellLabels,
							    const SIM_RawIndexField &materialCellLabels) const
{
    using SIM::FieldUtils::getFieldValue;
    using MGCellLabels = HDK::GeometricMultigridOperators::CellLabels;

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(materialCellLabels.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());

	if (boss->opInterrupt())
	    return;    

	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    // TODO: handle constant tiles
	    if (!vit.isTileConstant() ||
		vit.getValue() == MaterialLabels::LIQUID_CELL ||
		vit.getValue() == MaterialLabels::AIR_CELL)
	    {
		for (; !vit.atEnd(); vit.advance())
		{
		    UT_Vector3I cell(vit.x(), vit.y(), vit.z());

		    if (vit.getValue() == MaterialLabels::LIQUID_CELL)
			mgDomainCellLabels.setValue(cell, MGCellLabels::INTERIOR_CELL);
		    else if (vit.getValue() == MaterialLabels::AIR_CELL)
			mgDomainCellLabels.setValue(cell, MGCellLabels::DIRICHLET_CELL);
		    else
		    {
			assert(mgDomainCellLabels(cell) == MGCellLabels::EXTERIOR_CELL);
			assert(getFieldValue(materialCellLabels, cell) == MaterialLabels::SOLID_CELL);
		    }
		}
	    }
	}
    });

    mgDomainCellLabels.collapseAllTiles();
}

void
HDK_GeometricFreeSurfacePressureSolver::buildMGBoundaryWeights(UT_VoxelArray<SolveReal> &boundaryWeights,
								const SIM_RawField &cutCellWeights,
								const SIM_RawField &liquidSurface,
								const SIM_RawField &validFaces,
								const SIM_RawIndexField &materialCellLabels,
								const UT_VoxelArray<int> &mgDomainCellLabels,
								const int axis) const
{
    using SIM::FieldUtils::faceToCellMap;
    using SIM::FieldUtils::getFieldValue;
    using SIM::FieldUtils::setFieldValue;

    using MGCellLabels = HDK::GeometricMultigridOperators::CellLabels;

    UT_Interrupt *boss = UTgetInterrupt();

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

	    if (!vit.isTileConstant() ||
		vit.getValue() == HDK::Utilities::VALID_FACE)
	    {
		for (; !vit.atEnd(); vit.advance())
		{
		    if (vit.getValue() == HDK::Utilities::VALID_FACE)
		    {
			UT_Vector3I face(vit.x(), vit.y(), vit.z());

			SolveReal weight = getFieldValue(cutCellWeights, face);
			assert(weight > 0);

			UT_Vector3I backwardCell = faceToCellMap(face, axis, 0);
			UT_Vector3I forwardCell = faceToCellMap(face, axis, 1);

			auto backwardCellLabel = mgDomainCellLabels(backwardCell);
			auto forwardCellLabel = mgDomainCellLabels(forwardCell);

			if ((backwardCellLabel == MGCellLabels::INTERIOR_CELL && forwardCellLabel == MGCellLabels::DIRICHLET_CELL) ||
			    (backwardCellLabel == MGCellLabels::DIRICHLET_CELL && forwardCellLabel == MGCellLabels::INTERIOR_CELL))
			{
			    assert(getFieldValue(materialCellLabels, backwardCell) == MaterialLabels::AIR_CELL ||
				    getFieldValue(materialCellLabels, forwardCell) == MaterialLabels::AIR_CELL);

			    SolveReal phi0 = getFieldValue(liquidSurface, backwardCell);
			    SolveReal phi1 = getFieldValue(liquidSurface, forwardCell);

			    SolveReal theta = HDK::Utilities::computeGhostFluidWeight(phi0, phi1);
			    theta = SYSclamp(theta, .01, 1.);

			    weight /= theta;
			}

			boundaryWeights.setValue(face, weight);
		    }
		}
	    }
	}
    });
}

void
HDK_GeometricFreeSurfacePressureSolver::buildRHS(UT_VoxelArray<StoreReal> &rhsGrid,
						    const SIM_RawIndexField &materialCellLabels,
						    const SIM_VectorField &velocity,
						    const SIM_VectorField *solidVelocity,
						    const std::array<const SIM_RawField *, 3> &cutCellWeights,
						    const UT_VoxelArray<int> &mgDomainCellLabels,
						    const UT_Vector3I mgExpandedOffset) const
{
    using SIM::FieldUtils::cellToFaceMap;
    using SIM::FieldUtils::getFieldValue;
    using MGCellLabels = HDK::GeometricMultigridOperators::CellLabels;

    // Pre-expand all tiles that correspond to INTERIOR or BOUNDARY domain labels
    assert(rhsGrid.getVoxelRes() == mgDomainCellLabels.getVoxelRes());
    HDK::GeometricMultigridOperators::uncompressActiveGrid(rhsGrid, mgDomainCellLabels);

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(materialCellLabels.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());

	if (boss->opInterrupt())
	    return;

	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.isTileConstant() ||
		vit.getValue() == MaterialLabels::LIQUID_CELL)
	    {
		for (; !vit.atEnd(); vit.advance())
		{
		    if (vit.getValue() == MaterialLabels::LIQUID_CELL)
		    {
			UT_Vector3I cell(vit.x(), vit.y(), vit.z());

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

			UT_Vector3I expandedCell = cell + mgExpandedOffset;
			assert(!rhsGrid.getLinearTile(rhsGrid.indexToLinearTile(expandedCell[0],
										expandedCell[1],
										expandedCell[2]))->isConstant());
			assert(mgDomainCellLabels(expandedCell) == MGCellLabels::INTERIOR_CELL || 
				mgDomainCellLabels(expandedCell) == MGCellLabels::BOUNDARY_CELL);

			rhsGrid.setValue(expandedCell, divergence);
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
							    const UT_VoxelArray<int> &mgDomainCellLabels,
							    const UT_Vector3I &mgExpandedOffset) const
{
    using SIM::FieldUtils::getFieldValue;
    using MGCellLabels = HDK::GeometricMultigridOperators::CellLabels;

    // Pre-expand all tiles that correspond to INTERIOR or BOUNDARY domain labels
    assert(solutionGrid.getVoxelRes() == mgDomainCellLabels.getVoxelRes());
    HDK::GeometricMultigridOperators::uncompressActiveGrid(solutionGrid, mgDomainCellLabels);

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(materialCellLabels.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());

	if (boss->opInterrupt())
	    return;
	 
	for (int tileNumber = range.begin(); tileNumber != range.end(); ++tileNumber)
	{
	    vit.myTileStart = tileNumber;
	    vit.myTileEnd = tileNumber + 1;
	    vit.rewind();

	    if (!vit.isTileConstant() || vit.getValue() == MaterialLabels::LIQUID_CELL)
	    {
		for (vit.rewind(); !vit.atEnd(); vit.advance())
		{
		    if (vit.getValue() == MaterialLabels::LIQUID_CELL)
		    {
			UT_Vector3I cell(vit.x(), vit.y(), vit.z());

			UT_Vector3I expandedCell = cell + mgExpandedOffset;
			assert(!solutionGrid.getLinearTile(solutionGrid.indexToLinearTile(expandedCell[0],
											    expandedCell[1],
											    expandedCell[2]))->isConstant());
			
			assert(mgDomainCellLabels(expandedCell) == MGCellLabels::INTERIOR_CELL ||
				mgDomainCellLabels(expandedCell) == MGCellLabels::BOUNDARY_CELL);

			solutionGrid.setValue(expandedCell, getFieldValue(pressure, cell));
		    }
		}
	    }
	}
    });
}

void
HDK_GeometricFreeSurfacePressureSolver::applySolutionToPressure(SIM_RawField &pressure,
								const SIM_RawIndexField &materialCellLabels,
								const UT_VoxelArray<int> &mgDomainCellLabels,
								const UT_VoxelArray<StoreReal> &solutionGrid,
								const UT_Vector3I &mgExpandedOffset) const
{
    using SIM::FieldUtils::setFieldValue;
    using MGCellLabels = HDK::GeometricMultigridOperators::CellLabels;

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(materialCellLabels.field()->numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIteratorI vit;
	vit.setConstArray(materialCellLabels.field());

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
		    vit.getValue() == MaterialLabels::LIQUID_CELL)
		{
		    for (; !vit.atEnd(); vit.advance())
		    {
			if (vit.getValue() == MaterialLabels::LIQUID_CELL)
			{
			    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
			    UT_Vector3I expandedCell = cell + mgExpandedOffset;

			    assert(mgDomainCellLabels(expandedCell) == MGCellLabels::INTERIOR_CELL ||
				    mgDomainCellLabels(expandedCell) == MGCellLabels::BOUNDARY_CELL);

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
								const SIM_RawField &cutCellWeights,	
								const SIM_RawField &liquidSurface,
								const SIM_RawField &pressure,
								const SIM_RawField &validFaces,
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
		    for (; !vit.atEnd(); vit.advance())
		    {	      
			if (vit.getValue() == HDK::Utilities::VALID_FACE)
			{
			    UT_Vector3I face(vit.x(), vit.y(), vit.z());

			    UT_Vector3I backwardCell = faceToCellMap(face, axis, 0);
			    UT_Vector3I forwardCell = faceToCellMap(face, axis, 1);

			    if (backwardCell[axis] < 0 || forwardCell[axis] >= voxelRes[axis])
				continue;

			    exint backwardMaterial = getFieldValue(materialCellLabels, backwardCell);
			    exint forwardMaterial = getFieldValue(materialCellLabels, forwardCell);

			    assert(backwardMaterial == MaterialLabels::LIQUID_CELL || forwardMaterial >= MaterialLabels::LIQUID_CELL);
			    assert(getFieldValue(cutCellWeights, face) > 0);

			    SolveReal gradient = getFieldValue(pressure, forwardCell) - getFieldValue(pressure, backwardCell);

			    if (backwardMaterial != MaterialLabels::LIQUID_CELL ||
				forwardMaterial != MaterialLabels::LIQUID_CELL)
			    {
				if (backwardMaterial == MaterialLabels::LIQUID_CELL)
				    assert(forwardMaterial == MaterialLabels::AIR_CELL);
				else
				{
				    assert(backwardMaterial == MaterialLabels::AIR_CELL);
				    assert(forwardMaterial == MaterialLabels::LIQUID_CELL);
				}

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
	}
    });
}

void
HDK_GeometricFreeSurfacePressureSolver::computeResultingDivergence(UT_Array<SolveReal> &parallelAccumulatedDivergence,
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