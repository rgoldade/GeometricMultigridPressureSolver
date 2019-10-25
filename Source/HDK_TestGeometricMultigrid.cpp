#include "HDK_TestGeometricMultigrid.h"

#include <Eigen/Sparse>

#include <PRM/PRM_Include.h>

#include <SIM/SIM_DopDescription.h>
#include <SIM/SIM_FieldUtils.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_PRMShared.h>

#include <UT/UT_DSOVersion.h>
#include <UT/UT_ParallelUtil.h>
#include <UT/UT_PerfMonAutoEvent.h>


// TODO: remove stop watch
#include <UT/UT_StopWatch.h>

#include "HDK_GeometricCGPoissonSolver.h"
#include "HDK_GeometricMultigridOperators.h"
#include "HDK_GeometricMultigridPoissonSolver.h"

void initializeSIM(void *)
{
   IMPLEMENT_DATAFACTORY(HDK_TestGeometricMultigrid);
}

// Standard constructor, note that BaseClass was crated by the
// DECLARE_DATAFACTORY and provides an easy way to chain through
// the class hierarchy.
HDK_TestGeometricMultigrid::HDK_TestGeometricMultigrid(const SIM_DataFactory *factory)
    : BaseClass(factory)
{
}

HDK_TestGeometricMultigrid::~HDK_TestGeometricMultigrid()
{
}

const SIM_DopDescription* HDK_TestGeometricMultigrid::getDopDescription()
{
    static PRM_Name	theGridSizeName("gridSize", "Grid Size");
    static PRM_Default  theGridSizeDefault(64);

    static PRM_Name	theUseComplexDomainName("useComplexDomain", "Use Complex Domain");

    static PRM_Name	theUseSolidSphereName("useSolidSphere", "Use Solid Sphere");
    static PRM_Conditional    theUseSolidSphereDisable("{ useComplexDomain == 0 }");

    static PRM_Name	theUseRandomInitialGuessName("useRandomInitialGuess", "Use Random Initial Guess");

    static PRM_Name	theDeltaFunctionAmplitudeName("deltaFunctionAmplitude", "Delta Function Amplitude");
    static PRM_Default	theDeltaFunctionAmplitudeDefault(1000);

    static PRM_Conditional    theSolversParametersDisable("{ testSmoother == 0 testConjugateGradient == 0}");


    // Test Conjugate Gradient
    static PRM_Name	theTestConjugateGradientSeparatorName("testConjugateGradientSeparator", "Test Conjugate Gradient Separator");

    static PRM_Name	theTestConjugateGradientName("testConjugateGradient", "Test Conjugate Gradient");

    static PRM_Name	theUseMultigridPreconditionerName("useMultigridPreconditioner", "Use Multigrid preconditioner");

    static PRM_Name	theSolveCGGeometricallyName("solveCGGeometrically", "Solve CG Geometrically");

    static PRM_Name	theMultigridLevelsName("multigridLevels", "Multigrid Levels");

    static PRM_Name	theSolverToleranceName("solverTolerance", "Solver Tolerance");
    static PRM_Default	theSolverToleranceDefault(1E-5);

    static PRM_Name	theMaxSolverIterationsName("maxSolverIterations", "Max Solver Iterations");
    static PRM_Default	theMaxSolverDefault(1000);

    static PRM_Conditional    theTestConjugateGradientDisable("{ testConjugateGradient == 0 }");

    // Test symmetry
    static PRM_Name	theTestSymmetrySeparatorName("testSymmetrySeparator", "Test Symmetry Separator");

    static PRM_Name	theTestSymmetryName("testSymmetry", "Test Symmetry");


    // Test one level v-cycle parameters

    static PRM_Name	theTestOneLevelVCycleSeparatorName("testOneLevelVCycleSeparator", "One Level V-Cycle Separator");

    static PRM_Name	theTestOneLevelVCycleName("testOneLevelVCycle", "Test One Level V-cycle");

    // Test smoother parameters

    static PRM_Name	theSmootherSeparatorName("smootherSeparator", "Smoother Separator");

    static PRM_Name	theTestSmootherName("testSmoother", "Test Smoother");

    static PRM_Name	theMaxSmootherIterationsName("maxSmootherIterations", "Max Smoother Iterations");
    static PRM_Default	theMaxSmootherIterationsDefault(1000);

    static PRM_Name	theUseGaussSeidelSmoothingName("useGaussSeidelSmoothing", "Use Gauss Seidel Smoothing");

    static PRM_Conditional    theTestSmootherParameterDisable("{ testSmoother == 0 }");

    static PRM_Template	theTemplates[] =
    {
	PRM_Template(PRM_INT, 1, &theGridSizeName, &theGridSizeDefault),

	PRM_Template(PRM_TOGGLE, 1, &theUseComplexDomainName),

	PRM_Template(PRM_TOGGLE, 1, &theUseSolidSphereName, PRMzeroDefaults,
			0, 0, 0, 0, 1, 0, &theUseSolidSphereDisable),

	PRM_Template(PRM_TOGGLE, 1, &theUseRandomInitialGuessName, PRMzeroDefaults,
			0, 0, 0, 0, 1, 0, &theSolversParametersDisable),

	PRM_Template(PRM_FLT, 1, &theDeltaFunctionAmplitudeName, &theDeltaFunctionAmplitudeDefault,
			0, 0, 0, 0, 1, 0, &theSolversParametersDisable),

	// Conjugate gradient test parameters
	PRM_Template(PRM_SEPARATOR, 1, &theTestConjugateGradientSeparatorName),

	PRM_Template(PRM_TOGGLE, 1, &theTestConjugateGradientName),

	PRM_Template(PRM_TOGGLE, 1, &theUseMultigridPreconditionerName, PRMzeroDefaults,
			0, 0, 0, 0, 1, 0, &theTestConjugateGradientDisable),
	
	PRM_Template(PRM_TOGGLE, 1, &theSolveCGGeometricallyName, PRMzeroDefaults,
			0, 0, 0, 0, 1, 0, &theTestConjugateGradientDisable),

	PRM_Template(PRM_INT, 1, &theMultigridLevelsName, PRMfourDefaults,
			0, 0, 0, 0, 1, 0, &theTestConjugateGradientDisable),

	PRM_Template(PRM_FLT, 1, &theSolverToleranceName, &theSolverToleranceDefault,
			0, 0, 0, 0, 1, 0, &theTestConjugateGradientDisable),

	PRM_Template(PRM_INT, 1, &theMaxSolverIterationsName, &theMaxSolverDefault,
			0, 0, 0, 0, 1, 0, &theTestConjugateGradientDisable),

	// Symmetry test parameters
	PRM_Template(PRM_SEPARATOR, 1, &theTestSymmetrySeparatorName),

	PRM_Template(PRM_TOGGLE, 1, &theTestSymmetryName),

	// One level v-cycle parameters
	PRM_Template(PRM_SEPARATOR, 1, &theTestOneLevelVCycleSeparatorName),

	PRM_Template(PRM_TOGGLE, 1, &theTestOneLevelVCycleName),


	// Smoother test parameters
	PRM_Template(PRM_SEPARATOR, 1, &theSmootherSeparatorName),

	PRM_Template(PRM_TOGGLE, 1, &theTestSmootherName),

	PRM_Template(PRM_INT, 1, &theMaxSmootherIterationsName, &theMaxSmootherIterationsDefault,
			0, 0, 0, 0, 1, 0, &theTestSmootherParameterDisable),

	PRM_Template(PRM_TOGGLE, 1, &theUseGaussSeidelSmoothingName, PRMzeroDefaults,
			0, 0, 0, 0, 1, 0, &theTestSmootherParameterDisable),

	    PRM_Template()
    };

    static SIM_DopDescription theDopDescription(true,
						"HDK_TestGeometricMultigrid",
						"HDK Test Geometric Multigrid",
						"$OS",
						classname(),
						theTemplates);

    setGasDescription(theDopDescription);

    return &theDopDescription;
}

template<typename StoreReal>
fpreal
buildComplexDomain(UT_VoxelArray<int> &domainCellLabels,
		    std::array<UT_VoxelArray<StoreReal>, 3> &boundaryWeights,
		    const int gridSize,
		    const bool useSolidSphere)
{
    using HDK::GeometricMultigridOperators::CellLabels;
    using SIM::FieldUtils::cellToCellMap;
    using SIM::FieldUtils::faceToCellMap;

    assert(gridSize > 0);

    domainCellLabels.size(gridSize, gridSize, gridSize);
    domainCellLabels.constant(CellLabels::EXTERIOR_CELL);

    // Create an implicit grid [0,1]x[0,1]x[0,1]
    // with a dx of 1 / gridSize.
    UT_Interrupt *boss = UTgetInterrupt();

    const fpreal dx  = 1. / fpreal(gridSize);

    UTparallelForEachNumber(domainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIterator<int> vit;
	vit.setConstArray(&domainCellLabels);

	UT_VoxelTileIterator<int> vitt;

	if (boss->opInterrupt())
	    return;
	 
	for (int i = range.begin(); i != range.end(); ++i)
	{
	    vit.myTileStart = i;
	    vit.myTileEnd = i + 1;
	    vit.rewind();

	    if (!vit.atEnd())
	    {
		vitt.setTile(vit);

		for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		{
		    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
		    UT_Vector3 point = UT_Vector3(dx) * UT_Vector3(cell);

		    // Solid sphere at center of domain
		    if (useSolidSphere)
		    {
			const UT_Vector3 sphereCenter(.5);
			const fpreal sphereRadius = .125;

			if (distance2(point, sphereCenter) - sphereRadius * sphereRadius < 0)
			{
			    vitt.setValue(CellLabels::EXTERIOR_CELL);
			    return;			    
			}
		    }

		    // Sine wave for air-liquid boundary.
		    fpreal sdf = point[0] - .5 + .25 * SYSsin(2. * M_PI * point[1]);
		    if (sdf > 0)
			vitt.setValue(CellLabels::DIRICHLET_CELL);
		    else
			vitt.setValue(CellLabels::INTERIOR_CELL);
		}
	    }
	}
    });

    // TODO: use proper ghost fluid and cut-cell weights
    // Build boundary weights
    for (int axis : {0,1,2})
    {
	UT_Vector3I size(gridSize, gridSize, gridSize);
	++size[axis];
	boundaryWeights[axis].size(size[0], size[1], size[2]);
	boundaryWeights[axis].constant(0);

	UTparallelForEachNumber(boundaryWeights[axis].numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<StoreReal> vit(&boundaryWeights[axis]);
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
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());

			bool isInterior = false;
			bool isExterior = false;
			for (int direction : {0, 1})
			{
			    UT_Vector3I cell = faceToCellMap(face, axis, direction);

			    if (cell[axis] < 0 || cell[axis] >= domainCellLabels.getVoxelRes()[axis])
			    {
				isExterior = true;
				continue;
			    }

			    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
				isInterior = true;
			    else if (domainCellLabels(cell) == CellLabels::EXTERIOR_CELL)
				isExterior = true;
			}

			if (isInterior && !isExterior)
			    vitt.setValue(1);
		    }
		}
	    }
	});
    }
    return dx;
}

template<typename StoreReal>
fpreal
buildSimpleDomain(UT_VoxelArray<int> &domainCellLabels,
		    std::array<UT_VoxelArray<StoreReal>, 3> &boundaryWeights,
		    const int gridSize,
		    const int dirichletBand)
{
    using HDK::GeometricMultigridOperators::CellLabels;
    using SIM::FieldUtils::forEachVoxelRange;
    using SIM::FieldUtils::faceToCellMap;

    assert(gridSize > 0);
    assert(dirichletBand >= 0);

    domainCellLabels.size(gridSize, gridSize, gridSize);
    domainCellLabels.constant(CellLabels::EXTERIOR_CELL);

    // Set outer layers to DIRICHLET

    // Set bottom face
    UT_Vector3I start(0, 0, 0);
    UT_Vector3I end(gridSize, dirichletBand, gridSize);

    forEachVoxelRange(start, end, [&](const UT_Vector3I &cell)
    {
	domainCellLabels.setValue(cell, CellLabels::DIRICHLET_CELL);
    });

    // Set top face
    start = UT_Vector3I(0, gridSize - dirichletBand, 0);
    end = UT_Vector3I(gridSize, gridSize, gridSize);

    forEachVoxelRange(start, end, [&](const UT_Vector3I &cell)
    {
	domainCellLabels.setValue(cell, CellLabels::DIRICHLET_CELL);
    });

    // Set left face
    start = UT_Vector3I(0, 0, 0);
    end = UT_Vector3I(dirichletBand, gridSize, gridSize);

    forEachVoxelRange(start, end, [&](const UT_Vector3I &cell)
    {
	domainCellLabels.setValue(cell, CellLabels::DIRICHLET_CELL);
    });

    // Set right face
    start = UT_Vector3I(gridSize - dirichletBand, 0, 0);
    end = UT_Vector3I(gridSize, gridSize, gridSize);

    forEachVoxelRange(start, end, [&](const UT_Vector3I &cell)
    {
	domainCellLabels.setValue(cell, CellLabels::DIRICHLET_CELL);
    });

    // Set front face
    start = UT_Vector3I(0, 0, 0);
    end = UT_Vector3I(gridSize, gridSize, dirichletBand);

    forEachVoxelRange(start, end, [&](const UT_Vector3I &cell)
    {
	domainCellLabels.setValue(cell, CellLabels::DIRICHLET_CELL);
    });

    // Set back face
    start = UT_Vector3I(0, 0, gridSize - dirichletBand);
    end = UT_Vector3I(gridSize, gridSize, gridSize);

    forEachVoxelRange(start, end, [&](const UT_Vector3I &cell)
    {
	domainCellLabels.setValue(cell, CellLabels::DIRICHLET_CELL);
    });

    // Fill interior region as INTERIOR

    UT_Vector3I startCell = UT_Vector3I(dirichletBand, dirichletBand, dirichletBand);
    UT_Vector3I endCell = UT_Vector3I(gridSize - dirichletBand, gridSize - dirichletBand, gridSize - dirichletBand);

    UT_Interrupt *boss = UTgetInterrupt();

    UTparallelForEachNumber(domainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
    {
	UT_VoxelArrayIterator<int> vit;
	vit.setConstArray(&domainCellLabels);

	UT_VoxelTileIterator<int> vitt;

	if (boss->opInterrupt())
	    return;
	 
	for (int i = range.begin(); i != range.end(); ++i)
	{
	    vit.myTileStart = i;
	    vit.myTileEnd = i + 1;
	    vit.rewind();

	    if (!vit.atEnd())
	    {
		vitt.setTile(vit);

		for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		{
		    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());
		    if (cell[0] >= startCell[0] && cell[0] < endCell[0] &&
			cell[1] >= startCell[1] && cell[1] < endCell[1] &&
			cell[2] >= startCell[2] && cell[2] < endCell[2])
		    {
			vitt.setValue(CellLabels::INTERIOR_CELL);
		    }
		}
	    }
	}
    });
 
    // Build boundary weights
    for (int axis : {0,1,2})
    {
	UT_Vector3I size(gridSize, gridSize, gridSize);
	++size[axis];
	boundaryWeights[axis].size(size[0], size[1], size[2]);
	boundaryWeights[axis].constant(0);

	UTparallelForEachNumber(boundaryWeights[axis].numTiles(), [&](const UT_BlockedRange<int> &range)
	{
	    UT_VoxelArrayIterator<StoreReal> vit;
	    vit.setConstArray(&boundaryWeights[axis]);

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
		    vitt.setTile(vit);

		    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
		    {
			UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());

			bool isInterior = false;
			bool isExterior = false;
			for (int direction : {0, 1})
			{
			    UT_Vector3I cell = faceToCellMap(face, axis, direction);

			    if (cell[axis] < 0 || cell[axis] >= domainCellLabels.getVoxelRes()[axis])
			    {
				isExterior = true;
				continue;
			    }

			    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
				isInterior = true;
			    else if (domainCellLabels(cell) == CellLabels::EXTERIOR_CELL)
				isExterior = true;
			}

			if (isInterior && !isExterior)
			    vitt.setValue(1);
		    }
		}
	    }
	});
    }

    // Return dx
    return 1. / fpreal(gridSize);
}

bool HDK_TestGeometricMultigrid::solveGasSubclass(SIM_Engine &engine,
						    SIM_Object *obj,
						    SIM_Time time,
						    SIM_Time timestep)
{
    using namespace HDK::GeometricMultigridOperators;
    using SIM::FieldUtils::forEachVoxelRange;
    using SIM::FieldUtils::cellToCellMap;
    using SIM::FieldUtils::cellToFaceMap;

    using StoreReal = double;
    using SolveReal = double;

    const int gridSize = getGridSize();

    std::cout.precision(10);

    SolveReal dx;
    UT_Vector3I exteriorOffset;
    int mgLevels;

    UT_Interrupt *boss = UTgetInterrupt();

    UT_VoxelArray<int> domainCellLabels;
    std::array<UT_VoxelArray<StoreReal>, 3> boundaryWeights;    
    {
	std::cout << "  Build test domain" << std::endl;

	UT_VoxelArray<int> baseDomainCellLabels;
	std::array<UT_VoxelArray<StoreReal>, 3> baseBoundaryWeights;
	if (getUseComplexDomain())
	{
	    if (getUseSolidSphere())
		dx = buildComplexDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, true /* use solid sphere */);
	    else
		dx = buildComplexDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, false /* use solid sphere */);
	}
	else
	    dx = buildSimpleDomain(baseDomainCellLabels, baseBoundaryWeights, gridSize, 1 /*Dirichlet band*/);

	// Build expanded domain
	auto isExteriorCell = [](const int value) { return value == CellLabels::EXTERIOR_CELL; };
	auto isInteriorCell = [](const int value) { return value == CellLabels::INTERIOR_CELL; };
	auto isDirichletCell = [](const int value) { return value == CellLabels::DIRICHLET_CELL; };

	std::pair<UT_Vector3I, int> mgSettings = buildExpandedMGDomainLabels(domainCellLabels, baseDomainCellLabels, isExteriorCell, isInteriorCell, isDirichletCell);
	
	exteriorOffset = mgSettings.first;
	mgLevels = mgSettings.second;

	// Build expanded boundary weights
	for (int axis : {0,1,2})
	{
	    UT_Vector3I size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    ++size[axis];
	    boundaryWeights[axis].size(size[0], size[1], size[2]);
	    boundaryWeights[axis].constant(0);

	    buildExpandedBoundaryWeights(boundaryWeights[axis], baseBoundaryWeights[axis], domainCellLabels, exteriorOffset, axis);
	}

	// Build boundary cells
	setBoundaryDomainLabels(domainCellLabels, boundaryWeights);
    }

    assert(unitTestBoundaryCells(domainCellLabels));
    assert(unitTestExteriorCells(domainCellLabels));

    if (getTestConjugateGradient())
    {
	std::cout << "\n// Testing conjugate gradient" << std::endl;
	
	UT_VoxelArray<StoreReal> solutionGrid;
	solutionGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	solutionGrid.constant(0);
	
	if (getUseRandomInitialGuess())
	{
	    std::cout << "  Build random initial guess" << std::endl;

	    std::default_random_engine generator;
	    std::uniform_real_distribution<StoreReal> distribution(0, 1);
	
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
		    
		    UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for write */> solutionProbe;
		    solutionProbe.setArray(&solutionGrid);

		    if (!vit.isTileConstant() ||
			vit.getValue() == CellLabels::INTERIOR_CELL ||
			vit.getValue() == CellLabels::BOUNDARY_CELL)
		    {
			for (vit.rewind(); !vit.atEnd(); vit.advance())
			{
			    if (vit.getValue() == CellLabels::INTERIOR_CELL ||
				vit.getValue() == CellLabels::BOUNDARY_CELL)
			    {
				solutionProbe.setIndex(vit);
				solutionProbe.setValue(distribution(generator));

			    }
			}
		    }
		}
	    });
	}
	else std::cout << "  Use zero initial guess" << std::endl;
	
	UT_VoxelArray<StoreReal> rhsGrid;
	rhsGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	rhsGrid.constant(0);

	// Set delta function
	SolveReal deltaPercent = .1;
	UT_Vector3I deltaPoint = UT_Vector3I(deltaPercent * UT_Vector3(gridSize));

	{
	    SolveReal deltaAmplitude = getDeltaFunctionAmplitude();

	    const UT_Vector3I startCell = deltaPoint - UT_Vector3I(1);
	    const UT_Vector3I endCell = deltaPoint + UT_Vector3I(2);
	    UT_Vector3I sampleCell;

	    for (sampleCell[0] = startCell[0]; sampleCell[0] < endCell[0]; ++sampleCell[0])
		for (sampleCell[1] = startCell[1]; sampleCell[1] < endCell[1]; ++sampleCell[1])
		    for (sampleCell[2] = startCell[2]; sampleCell[2] < endCell[2]; ++sampleCell[2])
			rhsGrid.setValue(sampleCell + exteriorOffset, deltaAmplitude);
	}

	auto applyMatrixVectorMultiply = [&domainCellLabels, &boundaryWeights, dx](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	{
	    assert(destination.getVoxelRes() == source.getVoxelRes() &&
		    source.getVoxelRes() == domainCellLabels.getVoxelRes());

	    // Matrix-vector multiplication
	    HDK::GeometricMultigridOperators::applyPoissonMatrix<SolveReal>(destination, source, domainCellLabels, dx, &boundaryWeights);
	};

	auto applyDotProduct = [&domainCellLabels](const UT_VoxelArray<StoreReal> &grid0, const UT_VoxelArray<StoreReal> &grid1)
	{
	    assert(grid0.getVoxelRes() == grid1.getVoxelRes() &&
		    grid1.getVoxelRes() == domainCellLabels.getVoxelRes());

	    return HDK::GeometricMultigridOperators::dotProduct<SolveReal>(grid0, grid1, domainCellLabels);
	};

	auto getSquaredL2Norm = [&domainCellLabels](const UT_VoxelArray<StoreReal> &grid)
	{
	    assert(grid.getVoxelRes() == domainCellLabels.getVoxelRes());

	    return HDK::GeometricMultigridOperators::squaredL2Norm<SolveReal>(grid, domainCellLabels);
	};

	auto addToVector = [&domainCellLabels](UT_VoxelArray<StoreReal> &destination,
						const UT_VoxelArray<StoreReal> &source,
						const StoreReal scale)
	{
	    assert(destination.getVoxelRes() == source.getVoxelRes() &&
		    source.getVoxelRes() == domainCellLabels.getVoxelRes());

	    HDK::GeometricMultigridOperators::addToVector<SolveReal>(destination, source, scale, domainCellLabels);
	};

	auto addScaledVector = [&domainCellLabels](UT_VoxelArray<StoreReal> &destination,
						    const UT_VoxelArray<StoreReal> &unscaledSource,
						    const UT_VoxelArray<StoreReal> &scaledSource,
						    const StoreReal scale)
	{
	    assert(destination.getVoxelRes() == unscaledSource.getVoxelRes() &&
		    unscaledSource.getVoxelRes() == scaledSource.getVoxelRes() &&
		    scaledSource.getVoxelRes() == domainCellLabels.getVoxelRes());

	    HDK::GeometricMultigridOperators::addVectors<SolveReal>(destination, unscaledSource, scaledSource, scale, domainCellLabels);
	};

	if (getUseMultigridPreconditioner())
	{
	    UT_StopWatch timer;
	    timer.start();

	    UT_StopWatch prebuildtimer;
	    prebuildtimer.start();

	    // Pre-build multigrid preconditioner
	    HDK::GeometricMultigridPoissonSolver mgPreconditioner(domainCellLabels,
								    boundaryWeights,										
								    std::min(mgLevels, int(getMultigridLevels())),
								    dx,
								    true /* use Gauss Seidel smoother */);
	    
	    auto buildtime = prebuildtimer.stop();
	    std::cout << "  MG pre-build time: " << buildtime << std::endl;
	    prebuildtimer.clear();
	    prebuildtimer.start();

	    auto applyMultigridPreconditioner = [&mgPreconditioner, &domainCellLabels](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	    {
		assert(destination.getVoxelRes() == source.getVoxelRes() &&
			source.getVoxelRes() == domainCellLabels.getVoxelRes());

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
	    auto time = timer.stop();
	    std::cout << "  MG preconditioned CG solve time: " << time << std::endl;

	    // Compute residual
	    UT_VoxelArray<StoreReal> residualGrid;
	    residualGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    residualGrid.constant(0);

	    // Compute r = b - Ax
	    computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx, &boundaryWeights);

	    std::cout << "  L-infinity error: " << HDK::GeometricMultigridOperators::infNorm(residualGrid, domainCellLabels) << std::endl;
	    std::cout << "  Relative L-2: " << SYSsqrt(HDK::GeometricMultigridOperators::squaredL2Norm<SolveReal>(residualGrid, domainCellLabels) / HDK::GeometricMultigridOperators::squaredL2Norm<SolveReal>(rhsGrid, domainCellLabels)) << std::endl;
	}
	else
	{
	    if (getSolveCGGeometrically())
	    {
		UT_StopWatch timer;
		timer.start();

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

			UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, true /* test for writes */> diagonalProbe;
			diagonalProbe.setArray(&diagonalPrecondGrid);

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
#if !defined(NDEBUG)
					    UT_Vector3I cell(vit.x(), vit.y(), vit.z());
					    for (int axis : {0,1,2})
						for (int direction : {0,1})
						{
						    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
						    assert(boundaryWeights[axis](face) == 1);

						}
#endif
					    diagonalProbe.setIndex(vit);
					    diagonalProbe.setValue(1. / 6.);
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
					    diagonalProbe.setIndex(vit);
					    diagonalProbe.setValue(1. / diagonal);
					}
				    }
				}
			    }
			}
		    });
		}

		auto applyDiagonalPreconditioner = [&domainCellLabels, &diagonalPrecondGrid](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
		{
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

		auto time = timer.stop();
		std::cout << "  CG solve time: " << time << std::endl;

		// Compute residual
		UT_VoxelArray<StoreReal> residualGrid;
		residualGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
		residualGrid.constant(0);

		// Compute r = b - Ax
		computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx, &boundaryWeights);

		std::cout << "  L-infinity error: " << HDK::GeometricMultigridOperators::infNorm(residualGrid, domainCellLabels) << std::endl;
		std::cout << "  Relative L-2: " << HDK::GeometricMultigridOperators::l2Norm<SolveReal>(residualGrid, domainCellLabels) / HDK::GeometricMultigridOperators::l2Norm<SolveReal>(rhsGrid, domainCellLabels) << std::endl;
	    }
	    else
	    {
		exint interiorCellCount = 0;

		UT_VoxelArray<int> solverIndices;
		solverIndices.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);;
		solverIndices.constant(-1);

		// TODO: make parallel

		forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			solverIndices.setValue(cell, interiorCellCount++);
		});

		SolveReal gridScalar = 1. / (dx * dx);

		std::vector<Eigen::Triplet<StoreReal>> sparseElements;

		forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {

			SolveReal diagonal = 0;
			exint index = solverIndices(cell);
			assert(index >= 0);

			if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
			{
			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				    assert(domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
					    domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);

				    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				    assert(boundaryWeights[axis](face) == 1);

				    exint adjacentIndex = solverIndices(adjacentCell);
				    assert(adjacentIndex >= 0);

				    sparseElements.push_back(Eigen::Triplet<StoreReal>(index, adjacentIndex, -gridScalar));
				    diagonal += gridScalar;
				}
			}
			else if (domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			{
			    for (int axis : {0,1,2})
				for (int direction : {0,1})
				{
				    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < domainCellLabels.getVoxelRes()[axis]);

				    auto adjacentCellLabel = domainCellLabels(adjacentCell);
				    if (adjacentCellLabel == CellLabels::INTERIOR_CELL)
				    {
					UT_Vector3I face = cellToFaceMap(cell, axis, direction);

					assert(boundaryWeights[axis](face) == 1);

					exint adjacentIndex = solverIndices(adjacentCell);
					assert(adjacentIndex >= 0);

					sparseElements.push_back(Eigen::Triplet<StoreReal>(index, adjacentIndex, -gridScalar));
					diagonal += gridScalar;
				    }
				    else if (adjacentCellLabel == CellLabels::BOUNDARY_CELL)
				    {
					exint adjacentIndex = solverIndices(adjacentCell);
					assert(adjacentIndex >= 0);

					UT_Vector3I face = cellToFaceMap(cell, axis, direction);
					SolveReal weight = boundaryWeights[axis](face);

					sparseElements.push_back(Eigen::Triplet<StoreReal>(index, adjacentIndex, -gridScalar * weight));
					diagonal += gridScalar * weight;
				    }
				    else if (adjacentCellLabel == CellLabels::DIRICHLET_CELL)
				    {
					exint adjacentIndex = solverIndices(adjacentCell);
					assert(adjacentIndex == -1);

					UT_Vector3I face = cellToFaceMap(cell, axis, direction);
					SolveReal weight = boundaryWeights[axis](face);

					diagonal += gridScalar * weight;
				    }
				    else
				    {
					assert(adjacentCellLabel == CellLabels::EXTERIOR_CELL);
					UT_Vector3I face = cellToFaceMap(cell, axis, direction);
					assert(boundaryWeights[axis](face) == 0);
				    }
				}
			}
			assert(diagonal > 0);
			sparseElements.push_back(Eigen::Triplet<StoreReal>(index, index, diagonal));
		    }
		});
		
		using Vector = std::conditional<std::is_same<StoreReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;
		Vector rhs = Vector::Zero(interiorCellCount);

		forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = solverIndices(cell);
			assert(index >= 0);

			rhs(index) = rhsGrid(cell);
		    }
		});

		// Solve system
		Eigen::SparseMatrix<StoreReal> sparseMatrix(interiorCellCount, interiorCellCount);
		sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
		sparseMatrix.makeCompressed();

		UT_StopWatch timer;
		timer.start();

		Eigen::ConjugateGradient<Eigen::SparseMatrix<StoreReal>, Eigen::Upper | Eigen::Lower > solver(sparseMatrix);
		assert(solver.info() == Eigen::Success);

		solver.setTolerance(getSolverTolerance());

		Vector solution = solver.solve(rhs);

		auto time = timer.stop();
		std::cout << "  CG solve time: " << time << std::endl;

		Vector residual = sparseMatrix * solution;
		residual = rhs - residual;

		std::cout << "Solver iterations: " << solver.iterations() << std::endl;
		std::cout << "Solve error: " << solver.error() << std::endl;
		std::cout << "Re-computed residual: " << std::sqrt(residual.squaredNorm() / rhs.squaredNorm()) << std::endl;
	    }
	}
    }

    if (getTestSymmetry())
    {
	std::cout << "\n// Testing v-cycle symmetry" << std::endl;

	UT_VoxelArray<StoreReal> rhsA;
	rhsA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	rhsA.constant(0);

	UT_VoxelArray<StoreReal> rhsB;
	rhsB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	rhsB.constant(0);

	{
	    std::default_random_engine generator;
	    std::uniform_real_distribution<StoreReal> distribution(0, 1);

	    forEachVoxelRange(UT_Vector3I(exint(0)), rhsA.getVoxelRes(), [&](const UT_Vector3I &cell)
	    {
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
		    rhsA.setValue(cell, distribution(generator));
	    });

	    forEachVoxelRange(UT_Vector3I(exint(0)), rhsB.getVoxelRes(), [&](const UT_Vector3I &cell)
	    {
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
		    rhsB.setValue(cell, distribution(generator));
	    });
	}

	// Test Jacobi smoother
	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionA.constant(0);

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionB.constant(0);

	    UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(domainCellLabels, 2);

	    // Uncompress tiles on the boundary
	    uncompressBoundaryTiles(solutionA, boundaryCells);
	    uncompressBoundaryTiles(solutionB, boundaryCells);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionA, rhsA, domainCellLabels, boundaryCells, dx);

	    // Test Jacobi symmetry
	    interiorJacobiPoissonSmoother<SolveReal>(solutionA,
							rhsA,
							domainCellLabels,
							dx);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionA, rhsA, domainCellLabels, boundaryCells, dx);


	    boundaryJacobiPoissonSmoother<SolveReal>(solutionB, rhsB, domainCellLabels, boundaryCells, dx);

	    interiorJacobiPoissonSmoother<SolveReal>(solutionB,
							rhsB,
							domainCellLabels,
							dx);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionB, rhsB, domainCellLabels, boundaryCells, dx);

	    StoreReal dotA = dotProduct<SolveReal>(solutionA, rhsB, domainCellLabels);
	    StoreReal dotB = dotProduct<SolveReal>(solutionB, rhsA, domainCellLabels);

	    std::cout << "  Jacobi smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionA.constant(0);

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionB.constant(0);

	    UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(domainCellLabels, 2);

	    // Uncompress tiles on the boundary
	    uncompressBoundaryTiles(solutionA, boundaryCells);
	    uncompressBoundaryTiles(solutionB, boundaryCells);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							rhsA,
							domainCellLabels,
							boundaryCells,
							dx,
							&boundaryWeights);
	    // Test Jacobi symmetry
	    interiorJacobiPoissonSmoother<SolveReal>(solutionA,
							rhsA,
							domainCellLabels,
							dx);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							rhsA,
							domainCellLabels,
							boundaryCells,
							dx,
							&boundaryWeights);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							rhsB,
							domainCellLabels,
							boundaryCells,
							dx,
							&boundaryWeights);

	    interiorJacobiPoissonSmoother<SolveReal>(solutionB,
							rhsB,
							domainCellLabels,
							dx);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							rhsB,
							domainCellLabels,
							boundaryCells,
							dx,
							&boundaryWeights);

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, domainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, domainCellLabels);
	    
	    std::cout << "  Weighted Boundary Jacobi smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	// Test Gauss Seidel smoother
	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionA.constant(0);

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionB.constant(0);

	    UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(domainCellLabels, 2);

	    // Uncompress tiles on the boundary
	    uncompressBoundaryTiles(solutionA, boundaryCells);
	    uncompressBoundaryTiles(solutionB, boundaryCells);

	    for (int iteration = 0; iteration < 4; ++iteration)
	    {
		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionA,
								    rhsA,
								    domainCellLabels,
								    dx,
								    true /* smooth odd tiles */,
								    true /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionA,
								    rhsA,
								    domainCellLabels,
								    dx,
								    false /* smooth even tiles */,
								    true /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionA,
								    rhsA,
								    domainCellLabels,
								    dx,
								    false /* smooth even tiles */,
								    false /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionA,
								    rhsA,
								    domainCellLabels,
								    dx,
								    true /* smooth odd tiles */,
								    false /* iterate forward */);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);
	    }

	    for (int iteration = 0; iteration < 4; ++iteration)
	    {
		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionB,
								    rhsB,
								    domainCellLabels,
								    dx,
								    true /* smooth odd tiles */,
								    true /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionB,
								    rhsB,
								    domainCellLabels,
								    dx,
								    false /* smooth even tiles */,
								    true /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionB,
								    rhsB,
								    domainCellLabels,
								    dx,
								    false /* smooth even tiles */,
								    false /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionB,
								    rhsB,
								    domainCellLabels,
								    dx,
								    true /* smooth odd tiles */,
								    false /* iterate forward */);
		
		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);							    
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, domainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, domainCellLabels);

	    std::cout << "  Weighted Gauss Seidel smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    exint interiorCellCount = 0;

	    UT_VoxelArray<int> solverIndices;
	    solverIndices.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);;
	    solverIndices.constant(-1);

	    // TODO: make parallel

	    forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
	    {
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
		    domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    solverIndices.setValue(cell, interiorCellCount++);
	    });

	    SolveReal gridScalar = 1. / (dx * dx);

	    std::vector<Eigen::Triplet<StoreReal>> sparseElements;

	    forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
	    {
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
		    domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		{

		    SolveReal diagonal = 0;
		    exint index = solverIndices(cell);
		    assert(index >= 0);

		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
		    {
			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				assert(domainCellLabels(adjacentCell) == CellLabels::INTERIOR_CELL ||
					domainCellLabels(adjacentCell) == CellLabels::BOUNDARY_CELL);

				UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				assert(boundaryWeights[axis](face) == 1);

				exint adjacentIndex = solverIndices(adjacentCell);
				assert(adjacentIndex >= 0);

				sparseElements.push_back(Eigen::Triplet<StoreReal>(index, adjacentIndex, -gridScalar));
				diagonal += gridScalar;
			    }
		    }
		    else if (domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < domainCellLabels.getVoxelRes()[axis]);

				auto adjacentCellLabel = domainCellLabels(adjacentCell);
				if (adjacentCellLabel == CellLabels::INTERIOR_CELL)
				{
				    UT_Vector3I face = cellToFaceMap(cell, axis, direction);

				    assert(boundaryWeights[axis](face) == 1);

				    exint adjacentIndex = solverIndices(adjacentCell);
				    assert(adjacentIndex >= 0);

				    sparseElements.push_back(Eigen::Triplet<StoreReal>(index, adjacentIndex, -gridScalar));
				    diagonal += gridScalar;
				}
				else if (adjacentCellLabel == CellLabels::BOUNDARY_CELL)
				{
				    exint adjacentIndex = solverIndices(adjacentCell);
				    assert(adjacentIndex >= 0);

				    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				    SolveReal weight = boundaryWeights[axis](face);

				    sparseElements.push_back(Eigen::Triplet<StoreReal>(index, adjacentIndex, -gridScalar * weight));
				    diagonal += gridScalar * weight;
				}
				else if (adjacentCellLabel == CellLabels::DIRICHLET_CELL)
				{
				    exint adjacentIndex = solverIndices(adjacentCell);
				    assert(adjacentIndex == -1);

				    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				    SolveReal weight = boundaryWeights[axis](face);

				    diagonal += gridScalar * weight;
				}
				else
				{
				    assert(adjacentCellLabel == CellLabels::EXTERIOR_CELL);
				    UT_Vector3I face = cellToFaceMap(cell, axis, direction);
				    assert(boundaryWeights[axis](face) == 0);
				}
			    }
		    }
		    assert(diagonal > 0);
		    sparseElements.push_back(Eigen::Triplet<StoreReal>(index, index, diagonal));
		}
	    });
	    
	    // Solve system
	    Eigen::SparseMatrix<SolveReal> sparseMatrix(interiorCellCount, interiorCellCount);
	    sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
	    sparseMatrix.makeCompressed();

	    Eigen::SimplicialLDLT<Eigen::SparseMatrix<SolveReal>> solver;
	    solver.compute(sparseMatrix);
	    assert(solver.info() == Eigen::Success);

	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionA.constant(0);
	    
	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionB.constant(0);

	    using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

	    {
		Vector rhsVector = Vector::Zero(interiorCellCount);
		
		forEachVoxelRange(UT_Vector3I(exint(0)), rhsA.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = solverIndices(cell);
			assert(index >= 0);

			rhsVector(index) = rhsA(cell);
		    }
		});

		Vector solutionVector = solver.solve(rhsVector);

		forEachVoxelRange(UT_Vector3I(exint(0)), solutionA.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = solverIndices(cell);
			assert(index >= 0);

			solutionA.setValue(cell, solutionVector(index));
		    }
		});
	    }

	    {
		Vector rhsVector = Vector::Zero(interiorCellCount);
		
		forEachVoxelRange(UT_Vector3I(exint(0)), rhsB.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = solverIndices(cell);
			assert(index >= 0);

			rhsVector(index) = rhsB(cell);
		    }
		});

		Vector solutionVector = solver.solve(rhsVector);

		forEachVoxelRange(UT_Vector3I(exint(0)), solutionB.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = solverIndices(cell);
			assert(index >= 0);

			solutionB.setValue(cell, solutionVector(index));
		    }
		});
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, domainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, domainCellLabels);

	    std::cout << "  Direct solve symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    // Test down and up sampling
	    UT_VoxelArray<int> coarseDomainLabels = buildCoarseCellLabels(domainCellLabels);
	    
	    assert(unitTestBoundaryCells(coarseDomainLabels) && unitTestBoundaryCells(domainCellLabels));
	    assert(unitTestExteriorCells(coarseDomainLabels) && unitTestExteriorCells(domainCellLabels));
	    assert(unitTestCoarsening(coarseDomainLabels, domainCellLabels));

	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionA.constant(0);
	    
	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionB.constant(0);

	    UT_Vector3I coarseGridSize = coarseDomainLabels.getVoxelRes();
		
	    {
		UT_VoxelArray<StoreReal> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseRhs.constant(0);

		downsample<SolveReal>(coarseRhs, rhsA, coarseDomainLabels, domainCellLabels);
		upsampleAndAdd<SolveReal>(solutionA, coarseRhs, domainCellLabels, coarseDomainLabels);
	    }
	    {
		UT_VoxelArray<StoreReal> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseRhs.constant(0);

		downsample<SolveReal>(coarseRhs, rhsB, coarseDomainLabels, domainCellLabels);
		upsampleAndAdd<SolveReal>(solutionB, coarseRhs, domainCellLabels, coarseDomainLabels);
	    }		

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, domainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, domainCellLabels);

	    std::cout << "  Restriction/prolongation symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

	    // Test single level correction
	    UT_VoxelArray<int> coarseDomainLabels = buildCoarseCellLabels(domainCellLabels);

	    assert(unitTestBoundaryCells(coarseDomainLabels) && unitTestBoundaryCells(domainCellLabels));
	    assert(unitTestExteriorCells(coarseDomainLabels) && unitTestExteriorCells(domainCellLabels));
	    assert(unitTestCoarsening(coarseDomainLabels, domainCellLabels));

	    UT_Vector3I coarseGridSize = coarseDomainLabels.getVoxelRes();

	    exint interiorCellCount = 0;
	    UT_VoxelArray<exint> directSolverIndices;

	    directSolverIndices.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
	    directSolverIndices.constant(-1);

	    forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
	    {
		if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
		    coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
		    directSolverIndices.setValue(cell, interiorCellCount++);
	    });

	    std::vector<Eigen::Triplet<SolveReal>> sparseElements;

	    SolveReal gridScale = 1. / (dx * dx);
	    forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
	    {
		if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
		    coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
		{
		    SolveReal diagonal = 0;
		    int index = directSolverIndices(cell);
		    assert(index >= 0);
		    for (int axis : {0, 1, 2})
			for (int direction : {0, 1})
			{
			    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

			    assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < coarseGridSize[axis]);

			    auto adjacentCellLabels = coarseDomainLabels(adjacentCell);
			    if (adjacentCellLabels == CellLabels::INTERIOR_CELL ||
				adjacentCellLabels == CellLabels::BOUNDARY_CELL)
			    {
				exint adjacentIndex = directSolverIndices(adjacentCell);
				assert(adjacentIndex >= 0);

				sparseElements.push_back(Eigen::Triplet<SolveReal>(index, adjacentIndex, -gridScale));
				diagonal += gridScale;
			    }
			    else if (adjacentCellLabels == CellLabels::DIRICHLET_CELL)
				diagonal += gridScale;
			}

		    sparseElements.push_back(Eigen::Triplet<SolveReal>(index, index, diagonal));
		}
	    });

	    Eigen::SparseMatrix<SolveReal> sparseMatrix(interiorCellCount, interiorCellCount);
	    sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
	    sparseMatrix.makeCompressed();

	    Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolveReal>> solver(sparseMatrix);
	    assert(solver.info() == Eigen::Success);

	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionA.constant(0);
	    
	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionB.constant(0);

	    UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(domainCellLabels, 2);

	    // Uncompress tiles on the boundary
	    uncompressBoundaryTiles(solutionA, boundaryCells);
	    uncompressBoundaryTiles(solutionB, boundaryCells);

	    {
		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);

		interiorJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    domainCellLabels,
							    dx);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);

		// Compute new residual
		UT_VoxelArray<StoreReal> residual;
		residual.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
		residual.constant(0);

		computePoissonResidual<SolveReal>(residual, solutionA, rhsA, domainCellLabels, dx, &boundaryWeights);
		
		UT_VoxelArray<StoreReal> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseRhs.constant(0);
		
		downsample<SolveReal>(coarseRhs, residual, coarseDomainLabels, domainCellLabels);

		Vector rhsVector = Vector::Zero(interiorCellCount);

		// Copy to Eigen and direct solve
		forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
		{
		    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
			coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			rhsVector(index) = coarseRhs(cell);
		    }
		});

		UT_VoxelArray<StoreReal> coarseSolution;
		coarseSolution.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseSolution.constant(0);

		Vector solution = solver.solve(rhsVector);

		// Copy solution back
		forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
		{
		    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
			coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			coarseSolution.setValue(cell, solution(index));
		    }
		});

		upsampleAndAdd<SolveReal>(solutionA, coarseSolution, domainCellLabels, coarseDomainLabels);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);

		interiorJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    domainCellLabels,
							    dx);
		
		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);
	    }
	    {
		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);

		interiorJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    domainCellLabels,
							    dx);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);

		// Compute new residual
		UT_VoxelArray<StoreReal> residual;
		residual.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
		residual.constant(0);

		computePoissonResidual<SolveReal>(residual, solutionB, rhsB, domainCellLabels, dx, &boundaryWeights);
		
		UT_VoxelArray<StoreReal> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseRhs.constant(0);
		
		downsample<SolveReal>(coarseRhs, residual, coarseDomainLabels, domainCellLabels);

		Vector rhsVector = Vector::Zero(interiorCellCount);
		
		// Copy to Eigen and direct solve
		forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
		{
		    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
			coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			rhsVector(index) = coarseRhs(cell);
		    }
		});

		UT_VoxelArray<StoreReal> coarseSolution;
		coarseSolution.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseSolution.constant(0);

		Vector solution = solver.solve(rhsVector);

		// Copy solution back
		forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
		{
		    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
			coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			coarseSolution.setValue(cell, solution(index));
		    }
		});

		upsampleAndAdd<SolveReal>(solutionB, coarseSolution, domainCellLabels, coarseDomainLabels);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);

		interiorJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    domainCellLabels,
							    dx);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    domainCellLabels,
							    boundaryCells,
							    dx,
							    &boundaryWeights);
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, domainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, domainCellLabels);

	    std::cout << "  One level v cycle symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionA.constant(0);

	    {
		HDK::GeometricMultigridPoissonSolver mgSolver(domainCellLabels, boundaryWeights, mgLevels, dx, false);

		mgSolver.applyVCycle(solutionA, rhsA);
		mgSolver.applyVCycle(solutionA, rhsA, true);
		mgSolver.applyVCycle(solutionA, rhsA, true);
		mgSolver.applyVCycle(solutionA, rhsA, true);
	    }

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionB.constant(0);

	    {
		HDK::GeometricMultigridPoissonSolver mgSolver(domainCellLabels, boundaryWeights, mgLevels, dx, false);

		mgSolver.applyVCycle(solutionB, rhsB);
		mgSolver.applyVCycle(solutionB, rhsB, true);
		mgSolver.applyVCycle(solutionB, rhsB, true);
		mgSolver.applyVCycle(solutionB, rhsB, true);
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, domainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, domainCellLabels);

	    std::cout << "  Test multigrid poisson solver symmetry with Jacobi smoothing. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
    	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionA.constant(0);

	    {
		HDK::GeometricMultigridPoissonSolver mgSolver(domainCellLabels, boundaryWeights, mgLevels, dx, true);
		
		mgSolver.applyVCycle(solutionA, rhsA);
		mgSolver.applyVCycle(solutionA, rhsA, true);
		mgSolver.applyVCycle(solutionA, rhsA, true);
		mgSolver.applyVCycle(solutionA, rhsA, true);
	    }

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	    solutionB.constant(0);

	    {
		HDK::GeometricMultigridPoissonSolver mgSolver(domainCellLabels, boundaryWeights, mgLevels, dx, true);

		mgSolver.applyVCycle(solutionB, rhsB);
		mgSolver.applyVCycle(solutionB, rhsB, true);
		mgSolver.applyVCycle(solutionB, rhsB, true);
		mgSolver.applyVCycle(solutionB, rhsB, true);
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, domainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, domainCellLabels);

	    std::cout << "  Test multigrid poisson solver symmetry with Gauss Seidel smoothing. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
    }
    if (getTestOneLevelVCycle())
    {
	std::cout << "\n// Testing one level v-cycle" << std::endl;

	UT_VoxelArray<StoreReal> solutionGrid;
	solutionGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	solutionGrid.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	{
	    std::cout << "  Build initial guess" << std::endl;

	    // We have implicity built a grid sized [0,1]x[0,1]x[0,1]
	    // with a dx of 1/gridSize

	    UTparallelForEachNumber(domainCellLabels.numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<int> vit;
		vit.setConstArray(&domainCellLabels);
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
			if (!vit.isTileConstant() ||
			    vit.getValue() == CellLabels::INTERIOR_CELL ||
			    vit.getValue() == CellLabels::BOUNDARY_CELL)
			{
			    vitt.setTile(vit);

			    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			    {
				if (vitt.getValue() == CellLabels::INTERIOR_CELL ||
				    vitt.getValue() == CellLabels::BOUNDARY_CELL)
				{
				    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				    UT_Vector3 point = dx * UT_Vector3(cell);
				    solutionGrid.setValue(cell, std::sin(2 * M_PI * point[0]) * std::sin(2 * M_PI * point[1]) * std::sin(2 * M_PI * point[2]) +
								std::sin(4 * M_PI * point[0]) * std::sin(4 * M_PI * point[1]) * std::sin(4 * M_PI * point[1]));
				}
			    }
			}
		    }
		}
	    });
	}

	std::cout << "  Compute initial residual" << std::endl;

	// Implicitly set RHS to zero
	UT_VoxelArray<StoreReal> rhsGrid;
	rhsGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	rhsGrid.constant(0);

	UT_VoxelArray<StoreReal> residualGrid;
	residualGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	residualGrid.constant(0);

	computePoissonResidual<SolveReal>(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

	fpreal infNormError = infNorm(residualGrid, domainCellLabels);
	fpreal l2NormError = l2Norm<SolveReal>(residualGrid, domainCellLabels);

	std::cout << "L-infinity norm: " << infNormError << std::endl;
	std::cout << "L-2 norm: " << l2NormError << std::endl;

	std::cout << "  Apply one level of v-cycle" << std::endl;

	// Pre-build multigrid preconditioner
	HDK::GeometricMultigridPoissonSolver mgSolver(domainCellLabels, boundaryWeights, 2 /* levels in v-cycle */, dx, false /* use Jacobi smoothing */);

	mgSolver.applyVCycle(solutionGrid, rhsGrid, true /* use initial guess */);
	mgSolver.applyVCycle(solutionGrid, rhsGrid, true /* use initial guess */);
	mgSolver.applyVCycle(solutionGrid, rhsGrid, true /* use initial guess */);
	mgSolver.applyVCycle(solutionGrid, rhsGrid, true /* use initial guess */);

	std::cout << "  Compute test residual" << std::endl;

	computePoissonResidual<SolveReal>(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

	infNormError = infNorm(residualGrid, domainCellLabels);
	l2NormError = l2Norm<SolveReal>(residualGrid, domainCellLabels);

	std::cout << "L-infinity norm: " << infNormError << std::endl;
	std::cout << "L-2 norm: " << l2NormError << std::endl;
    }

    if (getTestSmoother())
    {
	std::cout << "\n// Testing smoother" << std::endl;
    
	UT_VoxelArray<StoreReal> rhsGrid;
	rhsGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	rhsGrid.constant(0);

	UT_VoxelArray<StoreReal> solutionGrid;
	solutionGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	solutionGrid.constant(0);

	UT_VoxelArray<StoreReal> residualGrid;
	residualGrid.size(domainCellLabels.getVoxelRes()[0], domainCellLabels.getVoxelRes()[1], domainCellLabels.getVoxelRes()[2]);
	residualGrid.constant(0);

	if (getUseRandomInitialGuess())
	{
	    std::cout << "  Build random initial guess" << std::endl;

	    std::default_random_engine generator;
	    std::uniform_real_distribution<StoreReal> distribution(0, 1);
	
	    forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
	    {
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
		    solutionGrid.setValue(cell, distribution(generator));
	    });
	}
	else std::cout << "  Use zero initial guess" << std::endl;

	std::cout << "  Set delta in RHS" << std::endl;

	// Set delta function
	SolveReal deltaPercent = .1;
	UT_Vector3I deltaPoint = UT_Vector3I(deltaPercent * UT_Vector3(gridSize));

	{
	    SolveReal deltaAmplitude = getDeltaFunctionAmplitude();

	    const UT_Vector3I startCell = deltaPoint - UT_Vector3I(1);
	    const UT_Vector3I endCell = deltaPoint + UT_Vector3I(2);
	    UT_Vector3I sampleCell;

	    for (sampleCell[0] = startCell[0]; sampleCell[0] < endCell[0]; ++sampleCell[0])
		for (sampleCell[1] = startCell[1]; sampleCell[1] < endCell[1]; ++sampleCell[1])
		    for (sampleCell[2] = startCell[2]; sampleCell[2] < endCell[2]; ++sampleCell[2])
			rhsGrid.setValue(sampleCell + exteriorOffset, deltaAmplitude);
	}

	UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(domainCellLabels, 2);

	using namespace HDK::GeometricMultigridOperators;

	// Uncompress tiles on the boundary
	uncompressBoundaryTiles(solutionGrid, boundaryCells);

	// Apply smoother
	const int maxSmootherIterations = getMaxSmootherIterations();
   
	const bool useGaussSeidelSmoothing = getUseGaussSeidelSmoothing();

	int iteration = 0;
	SolveReal solvetime = 0;
	SolveReal boundarySolveTime = 0;
	SolveReal count = 0;
	for (; iteration < maxSmootherIterations; ++iteration)
	{
	    if (useGaussSeidelSmoothing)
	    {
		UT_StopWatch timer;
		timer.start();

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionGrid, rhsGrid, domainCellLabels, dx, true /*smooth odd tiles*/, true /*smooth forwards*/);
		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionGrid, rhsGrid, domainCellLabels, dx, false /*smooth even tiles*/, true /*smooth forwards*/);
		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionGrid, rhsGrid, domainCellLabels, dx, false /*smooth even tiles*/, false /*smooth backwards*/);
		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionGrid, rhsGrid, domainCellLabels, dx, true /*smooth odd tiles*/, false /*smooth backwards*/);
		
		solvetime += timer.stop();
		++count;
		timer.clear();
		timer.start();

	    }
	    else
	    {
		UT_StopWatch timer;
		timer.start();

		interiorJacobiPoissonSmoother<SolveReal>(solutionGrid, rhsGrid, domainCellLabels, dx);

		solvetime += timer.stop();
		++count;
		timer.clear();
		timer.start();
	    }

	    {
		UT_StopWatch timer;
		timer.start();

		// Boundary smoothing
		boundaryJacobiPoissonSmoother<SolveReal>(solutionGrid, rhsGrid, domainCellLabels, boundaryCells, dx, &boundaryWeights);

		boundarySolveTime += timer.stop();
		timer.clear();
		timer.start();
	    }

	    computePoissonResidual<SolveReal>(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx, &boundaryWeights);

	    fpreal infNormError = infNorm(residualGrid, domainCellLabels);
	    fpreal l2NormError = l2Norm<SolveReal>(residualGrid, domainCellLabels);

	    std::cout << "Iteration: " << iteration << std::endl;
	    std::cout << "L-infinity norm: " << infNormError << std::endl;
	    std::cout << "L-2 norm: " << l2NormError << std::endl;
	}

	std::cout << "Interior smoother time: " << solvetime / count << std::endl;
	std::cout << "Boundary smoother time: " << boundarySolveTime / count << std::endl;
    }

    return true;
}