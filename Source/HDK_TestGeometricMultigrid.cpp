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

using namespace HDK::GeometricMultigridOperators;
using namespace SIM::FieldUtils;

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

UT_VoxelArray<int>
buildComplexDomain(const int gridSize,
		    const bool useSolidSphere,
		    const fpreal sphereRadius = .1)
{
    UT_VoxelArray<int> domainCellLabels;
    domainCellLabels.size(gridSize, gridSize, gridSize);
    domainCellLabels.constant(CellLabels::EXTERIOR_CELL);

    // Create an implicit grid [0,1]x[0,1]x[0,1]
    // with a dx of 1 / gridSize.

    UT_Vector3 sphereCenter(.5);
    UT_Vector3 dx(1. / fpreal(gridSize));

    forEachVoxelRange(UT_Vector3I(1), UT_Vector3I(gridSize) - UT_Vector3I(1), [&](const UT_Vector3I &cell)
    {
	UT_Vector3 point = dx * UT_Vector3(cell);

	// Solid sphere at center of domain
	if (useSolidSphere)
	{
	    if (distance2(point, sphereCenter) - sphereRadius * sphereRadius < 0)
	    {
		domainCellLabels.setValue(cell, CellLabels::EXTERIOR_CELL);
		return;			    
	    }
	}

	// Sine wave for air-liquid boundary.
	fpreal sdf = point[0] - .5 + .25 * SYSsin(2. * M_PI * point[1]);
	if (sdf > 0)
	    domainCellLabels.setValue(cell, CellLabels::DIRICHLET_CELL);
	else
	    domainCellLabels.setValue(cell, CellLabels::INTERIOR_CELL);
    });

    // Set boundary cells
    forEachVoxelRange(UT_Vector3I(1), UT_Vector3I(gridSize) - UT_Vector3I(1), [&](const UT_Vector3I &cell)
    {
	if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	{
	    for (int axis : {0,1,2})
		for (int direction : {0,1})
		{
		    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

		    if (domainCellLabels(adjacentCell) == DIRICHLET_CELL ||
			domainCellLabels(adjacentCell) == EXTERIOR_CELL)
		    {
			domainCellLabels.setValue(cell, CellLabels::BOUNDARY_CELL);
			return;
		    }
		}
	}
    });

    return domainCellLabels;
}

UT_VoxelArray<int>
buildSimpleDomain(const int gridSize,
		    const int dirichletBand)
{
    assert(gridSize % 2 == 0);
    assert(gridSize > 0);
    assert(dirichletBand >= 0);

    UT_VoxelArray<int> domainCellLabels;
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

    start = UT_Vector3I(dirichletBand, dirichletBand, dirichletBand);
    end = UT_Vector3I(gridSize - dirichletBand, gridSize - dirichletBand, gridSize - dirichletBand);

    forEachVoxelRange(start, end, [&](const UT_Vector3I &cell)
    {
	domainCellLabels.setValue(cell, CellLabels::INTERIOR_CELL);
    });

    // Set boundary cells

    forEachVoxelRange(UT_Vector3I(1), UT_Vector3I(gridSize) - UT_Vector3I(1), [&](const UT_Vector3I &cell)
    {
	if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	{
	    for (int axis : {0,1,2})
		for (int direction : {0,1})
		{
		    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

		    if (domainCellLabels(adjacentCell) == DIRICHLET_CELL ||
			domainCellLabels(adjacentCell) == EXTERIOR_CELL)
		    {
			domainCellLabels.setValue(cell, CellLabels::BOUNDARY_CELL);
			return;
		    }
		}
	}
    });
    
    return domainCellLabels;
}

bool HDK_TestGeometricMultigrid::solveGasSubclass(SIM_Engine &engine,
						    SIM_Object *obj,
						    SIM_Time time,
						    SIM_Time timestep)
{
    using StoreReal = float;
    using SolveReal = double;

    const int gridSize = getGridSize();

    std::cout << "  Build test domain" << std::endl;

    fpreal dx = 1. / fpreal(gridSize);

    UT_VoxelArray<int> domainCellLabels;
    if (getUseComplexDomain())
    {
	if (getUseSolidSphere())
	    domainCellLabels = buildComplexDomain(gridSize, true /* use solid sphere */, .125 /* sphere radius */);
	else
	    domainCellLabels = buildComplexDomain(gridSize, false /* use solid sphere */);
    }
    else
	domainCellLabels = buildSimpleDomain(gridSize, 1 /*Dirichlet band*/);

    assert(unitTestBoundaryCells(domainCellLabels));

    if (getTestConjugateGradient())
    {
	std::cout << "\n// Testing conjugate gradient" << std::endl;
	
	UT_VoxelArray<StoreReal> solutionGrid;
	solutionGrid.size(gridSize, gridSize, gridSize);
	solutionGrid.constant(0);
	
	if (getUseRandomInitialGuess())
	{
	    std::cout << "  Build random initial guess" << std::endl;

	    std::default_random_engine generator;
	    std::uniform_real_distribution<StoreReal> distribution(0, 1);
	
	    forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
	    {
		if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
		    domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    solutionGrid.setValue(cell, distribution(generator));

	    });
	}
	else std::cout << "  Use zero initial guess" << std::endl;
	
	UT_VoxelArray<StoreReal> rhsGrid;
	rhsGrid.size(gridSize, gridSize, gridSize);
	rhsGrid.constant(0);

	// Set delta function
	fpreal32 deltaPercent = .1;
	UT_Vector3I deltaPoint = UT_Vector3I(deltaPercent * UT_Vector3(gridSize));

	{
	    fpreal32 deltaAmplitude = getDeltaFunctionAmplitude();

	    const UT_Vector3I startCell = deltaPoint - UT_Vector3I(1);
	    const UT_Vector3I endCell = deltaPoint + UT_Vector3I(2);
	    UT_Vector3I sampleCell;

	    for (sampleCell[0] = startCell[0]; sampleCell[0] < endCell[0]; ++sampleCell[0])
		for (sampleCell[1] = startCell[1]; sampleCell[1] < endCell[1]; ++sampleCell[1])
		    for (sampleCell[2] = startCell[2]; sampleCell[2] < endCell[2]; ++sampleCell[2])
			rhsGrid.setValue(sampleCell, deltaAmplitude);
	}

	// Build dummy weights
	std::array<UT_VoxelArray<StoreReal>, 3> dummyWeights;
	for (int axis : {0,1,2})
	{
	    UT_Vector3I size(gridSize, gridSize, gridSize);
	    ++size[axis];
	    dummyWeights[axis].size(size[0], size[1], size[2]);
	    dummyWeights[axis].constant(0);

	    forEachVoxelRange(UT_Vector3I(exint(0)), dummyWeights[axis].getVoxelRes(), [&](const UT_Vector3I &face)
	    {
		bool isInterior = false;
		bool isExterior = false;
		for (int direction : {0, 1})
		{
		    UT_Vector3I cell = faceToCellMap(face, axis, direction);

		    if (cell[axis] < 0 || cell[axis] >= domainCellLabels.getVoxelRes()[axis])
			continue;

		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			isInterior = true;
		    else if (domainCellLabels(cell) == CellLabels::EXTERIOR_CELL)
			isExterior = true;
		}

		if (isInterior && !isExterior)
		    dummyWeights[axis].setValue(face, 1);
	    });
	}

	auto MatrixVectorMultiply = [&domainCellLabels, &dummyWeights, dx](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	{
	    // Matrix-vector multiplication
	    HDK::GeometricMultigridOperators::applyPoissonMatrix<SolveReal>(destination, source, domainCellLabels, dx, &dummyWeights);
	};

	auto DotProduct = [&domainCellLabels](const UT_VoxelArray<StoreReal> &grid0, const UT_VoxelArray<StoreReal> &grid1) -> fpreal32
	{
	    assert(grid0.getVoxelRes() == grid1.getVoxelRes());
	    return HDK::GeometricMultigridOperators::dotProduct<SolveReal>(grid0, grid1, domainCellLabels);
	};

	auto SquaredL2Norm = [&domainCellLabels](const UT_VoxelArray<StoreReal> &grid) -> fpreal32
	{
	    return HDK::GeometricMultigridOperators::squaredL2Norm<SolveReal>(grid, domainCellLabels);
	};

	auto AddToVector = [&domainCellLabels](UT_VoxelArray<StoreReal> &destination,
						const UT_VoxelArray<StoreReal> &source,
						const fpreal32 scale)
	{
	    HDK::GeometricMultigridOperators::addToVector<SolveReal>(destination, source, scale, domainCellLabels);
	};

	auto AddScaledVector = [&domainCellLabels](UT_VoxelArray<StoreReal> &destination,
						    const UT_VoxelArray<StoreReal> &unscaledSource,
						    const UT_VoxelArray<StoreReal> &scaledSource,
						    const fpreal32 scale)
	{
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
								    getMultigridLevels(),
								    dx,
								    3 /* boundary width */,
								    3 /* boundary smoother iterations */,
								    true /* use Gauss Seidel smoother */);
	    
	    auto buildtime = prebuildtimer.stop();
	    std::cout << "  MG pre-build time: " << buildtime << std::endl;
	    prebuildtimer.clear();
	    prebuildtimer.start();

	    mgPreconditioner.setBoundaryWeights(dummyWeights);

	    buildtime = prebuildtimer.stop();
	    std::cout << "  MG pre-build set weights time: " << buildtime << std::endl;

	    auto MultigridPreconditioner = [&mgPreconditioner](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
	    {
		assert(destination.getVoxelRes() == source.getVoxelRes());
		mgPreconditioner.applyVCycle(destination, source);
	    };

	    HDK::solveGeometricConjugateGradient(solutionGrid,
						    rhsGrid,
						    MatrixVectorMultiply,
						    MultigridPreconditioner,
						    DotProduct,
						    SquaredL2Norm,
						    AddToVector,
						    AddScaledVector,
						    fpreal32(getSolverTolerance()),
						    getMaxSolverIterations());
	    auto time = timer.stop();
	    std::cout << "  MG preconditioned CG solve time: " << time << std::endl;

	    // Compute residual
	    UT_VoxelArray<StoreReal> residualGrid;
	    residualGrid.size(gridSize, gridSize, gridSize);
	    residualGrid.constant(0);

	    // Compute r = b - Ax
	    MatrixVectorMultiply(residualGrid, solutionGrid);
	    AddScaledVector(residualGrid, rhsGrid, residualGrid, -1);

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
		diagonalPrecondGrid.size(gridSize, gridSize, gridSize);
		diagonalPrecondGrid.constant(0);

		forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    fpreal gridScalar = 1. / (dx * dx);
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			fpreal diagonal = 0;
			for (int axis : {0, 1, 2})
			    for (int direction : {0, 1})
			    {
				UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				if (adjacentCell[axis] < 0 || adjacentCell[axis] >= domainCellLabels.getVoxelRes()[axis])
				    continue;

				UT_Vector3I face = cellToFaceMap(cell, axis, direction);

				if (domainCellLabels(adjacentCell) == CellLabels::DIRICHLET_CELL)
				{
				    diagonal += dummyWeights[axis](face);
				}
				else ++diagonal;
			    }
			diagonal *= gridScalar;
			diagonalPrecondGrid.setValue(cell, 1. / diagonal);
		    }
		});
	    
		auto DiagonalPreconditioner = [&domainCellLabels, &diagonalPrecondGrid](UT_VoxelArray<StoreReal> &destination, const UT_VoxelArray<StoreReal> &source)
		{
		    assert(destination.getVoxelRes() == source.getVoxelRes());

		    const int tileCount = domainCellLabels.numTiles();

		    UT_Interrupt *boss = UTgetInterrupt();

		    UTparallelForEachNumber(tileCount, [&](const UT_BlockedRange<int> &range)
		    {
			UT_VoxelArrayIterator<int> vit;
			vit.setConstArray(&domainCellLabels);

			UT_VoxelProbe<StoreReal, false /* no read */, true /* write */, false> destinationProbe;
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
				SolveReal localSqr = 0;

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
							MatrixVectorMultiply,
							DiagonalPreconditioner,
							DotProduct,
							SquaredL2Norm,
							AddToVector,
							AddScaledVector,
							fpreal32(getSolverTolerance()),
							getMaxSolverIterations());

		auto time = timer.stop();
		std::cout << "  CG solve time: " << time << std::endl;

		// Compute residual
		UT_VoxelArray<StoreReal> residualGrid;
		residualGrid.size(gridSize, gridSize, gridSize);
		residualGrid.constant(0);

		// Compute r = b - Ax
		MatrixVectorMultiply(residualGrid, solutionGrid);
		AddScaledVector(residualGrid, rhsGrid, residualGrid, -1);

		std::cout << "  L-infinity error: " << HDK::GeometricMultigridOperators::infNorm(residualGrid, domainCellLabels) << std::endl;
		std::cout << "  Relative L-2: " << HDK::GeometricMultigridOperators::l2Norm<SolveReal>(residualGrid, domainCellLabels) / HDK::GeometricMultigridOperators::l2Norm<SolveReal>(rhsGrid, domainCellLabels) << std::endl;
	    }
	    else
	    {
		exint interiorCellCount = 0;

		UT_VoxelArray<int> solverIndices;
		solverIndices.size(gridSize, gridSize, gridSize);
		solverIndices.constant(-1);

		forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			solverIndices.setValue(cell, interiorCellCount++);
		});

		fpreal32 gridScalar = 1.; // / (dx * dx);

		std::vector<Eigen::Triplet<StoreReal>> sparseElements;

		forEachVoxelRange(UT_Vector3I(exint(0)), domainCellLabels.getVoxelRes(), [&](const UT_Vector3I &cell)
		{
		    if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			fpreal32 diagonal = 0;
			exint index = solverIndices(cell);
			assert(index >= 0);

			for (int axis : {0,1,2})
			    for (int direction : {0,1})
			    {
				UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

				assert(adjacentCell[axis] >= 0 && adjacentCell[axis] < domainCellLabels.getVoxelRes()[axis]);

				auto adjacentCellLabel = domainCellLabels(adjacentCell);
				if (adjacentCellLabel == CellLabels::INTERIOR_CELL ||
				    adjacentCellLabel == CellLabels::BOUNDARY_CELL)
				{
				    exint adjacentIndex = solverIndices(adjacentCell);
				    assert(adjacentIndex >= 0);

				    sparseElements.push_back(Eigen::Triplet<StoreReal>(index, adjacentIndex, -gridScalar));
				    diagonal += gridScalar;
				}
				else if (adjacentCellLabel == CellLabels::DIRICHLET_CELL)
				    diagonal += gridScalar;
			    }
			sparseElements.push_back(Eigen::Triplet<StoreReal>(index, index, diagonal));
		    }
		});

		Eigen::VectorXf rhs = Eigen::VectorXf::Zero(interiorCellCount);

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

		Eigen::VectorXf solution = solver.solve(rhs);

		auto time = timer.stop();
		std::cout << "  CG solve time: " << time << std::endl;


		Eigen::VectorXf residual = sparseMatrix * solution;
		residual = rhs - residual;

		std::cout << "Solver iterations: " << solver.iterations() << std::endl;
		std::cout << "Solve error: " << solver.error() << std::endl;
		std::cout << "Re-computed residual: " << residual.lpNorm<2>() / rhs.lpNorm<2>() << std::endl;
	    }
	}
    }

    if (getTestSymmetry())
    {
	std::cout.precision(10);

	std::cout << "\n// Testing v-cycle symmetry" << std::endl;
	
	int expandedOffset = 2;
	int expandedGridSize = gridSize + 2 * expandedOffset;

	// Build expanded domain cell labels
	UT_VoxelArray<int> expandedDomainCellLabels;
	expandedDomainCellLabels.size(expandedGridSize, expandedGridSize, expandedGridSize);
	expandedDomainCellLabels.constant(CellLabels::EXTERIOR_CELL);

	forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(gridSize)), [&](const UT_Vector3I &cell)
	{
	    expandedDomainCellLabels.setValue(cell + expandedOffset, domainCellLabels(cell));
	});

	UT_VoxelArray<StoreReal> rhsA;
	rhsA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	rhsA.constant(0);

	UT_VoxelArray<StoreReal> rhsB;
	rhsB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	rhsB.constant(0);

	{
	    std::default_random_engine generator;
	    std::uniform_real_distribution<StoreReal> distribution(0, 1);

	    forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	    {
		if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
		    rhsA.setValue(cell, distribution(generator));
	    });

	    forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	    {
		if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
		    rhsB.setValue(cell, distribution(generator));
	    });
	}

	// Build dummy weights
	std::array<UT_VoxelArray<StoreReal>, 3> dummyWeights;
	for (int axis : {0,1,2})
	{
	    UT_Vector3I size(expandedGridSize, expandedGridSize, expandedGridSize);
	    ++size[axis];
	    dummyWeights[axis].size(size[0], size[1], size[2]);
	    dummyWeights[axis].constant(0);

	    forEachVoxelRange(UT_Vector3I(exint(0)), dummyWeights[axis].getVoxelRes(), [&](const UT_Vector3I &face)
	    {
		bool isInterior = false;
		bool isExterior = false;
		for (int direction : {0, 1})
		{
		    UT_Vector3I cell = faceToCellMap(face, axis, direction);

		    if (cell[axis] < 0 || cell[axis] >= expandedDomainCellLabels.getVoxelRes()[axis])
			continue;

		    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			expandedDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
			isInterior = true;
		    else if (expandedDomainCellLabels(cell) == CellLabels::EXTERIOR_CELL)
			isExterior = true;
		}

		if (isInterior && !isExterior)
		    dummyWeights[axis].setValue(face, 1);
	    });
	}

	// Test Jacobi smoother
	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionA.constant(0);

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionB.constant(0);

	    UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(expandedDomainCellLabels, 2);

	    // Uncompress tiles on the boundary
	    uncompressBoundaryTiles(solutionA, boundaryCells);
	    uncompressBoundaryTiles(solutionB, boundaryCells);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionA, rhsA, expandedDomainCellLabels, boundaryCells, dx);

	    // Test Jacobi symmetry
	    interiorJacobiPoissonSmoother<SolveReal>(solutionA,
							rhsA,
							expandedDomainCellLabels,
							dx);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionA, rhsA, expandedDomainCellLabels, boundaryCells, dx);


	    boundaryJacobiPoissonSmoother<SolveReal>(solutionB, rhsB, expandedDomainCellLabels, boundaryCells, dx);

	    interiorJacobiPoissonSmoother<SolveReal>(solutionB,
							rhsB,
							expandedDomainCellLabels,
							dx);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionB, rhsB, expandedDomainCellLabels, boundaryCells, dx);

	    StoreReal dotA = dotProduct<SolveReal>(solutionA, rhsB, expandedDomainCellLabels);
	    StoreReal dotB = dotProduct<SolveReal>(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Jacobi smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionA.constant(0);

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionB.constant(0);

	    UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(expandedDomainCellLabels, 2);

	    // Uncompress tiles on the boundary
	    uncompressBoundaryTiles(solutionA, boundaryCells);
	    uncompressBoundaryTiles(solutionB, boundaryCells);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							rhsA,
							expandedDomainCellLabels,
							boundaryCells,
							dx,
							&dummyWeights);
	    // Test Jacobi symmetry
	    interiorJacobiPoissonSmoother<SolveReal>(solutionA,
							rhsA,
							expandedDomainCellLabels,
							dx);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							rhsA,
							expandedDomainCellLabels,
							boundaryCells,
							dx,
							&dummyWeights);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							rhsB,
							expandedDomainCellLabels,
							boundaryCells,
							dx,
							&dummyWeights);

	    interiorJacobiPoissonSmoother<SolveReal>(solutionB,
							rhsB,
							expandedDomainCellLabels,
							dx);

	    boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							rhsB,
							expandedDomainCellLabels,
							boundaryCells,
							dx,
							&dummyWeights);

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, expandedDomainCellLabels);
	    
	    std::cout << "  Weighted Boundary Jacobi smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	// Test Gauss Seidel smoother
	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionA.constant(0);

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionB.constant(0);

	    UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(expandedDomainCellLabels, 2);

	    // Uncompress tiles on the boundary
	    uncompressBoundaryTiles(solutionA, boundaryCells);
	    uncompressBoundaryTiles(solutionB, boundaryCells);

	    for (int iteration = 0; iteration < 4; ++iteration)
	    {
		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    expandedDomainCellLabels,
							    boundaryCells,
							    dx,
							    &dummyWeights);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionA,
								    rhsA,
								    expandedDomainCellLabels,
								    dx,
								    true /* smooth odd tiles */,
								    true /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionA,
								    rhsA,
								    expandedDomainCellLabels,
								    dx,
								    false /* smooth even tiles */,
								    true /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionA,
								    rhsA,
								    expandedDomainCellLabels,
								    dx,
								    false /* smooth even tiles */,
								    false /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionA,
								    rhsA,
								    expandedDomainCellLabels,
								    dx,
								    true /* smooth odd tiles */,
								    false /* iterate forward */);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    expandedDomainCellLabels,
							    boundaryCells,
							    dx,
							    &dummyWeights);
	    }

	    for (int iteration = 0; iteration < 4; ++iteration)
	    {
		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    expandedDomainCellLabels,
							    boundaryCells,
							    dx,
							    &dummyWeights);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionB,
								    rhsB,
								    expandedDomainCellLabels,
								    dx,
								    true /* smooth odd tiles */,
								    true /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionB,
								    rhsB,
								    expandedDomainCellLabels,
								    dx,
								    false /* smooth even tiles */,
								    true /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionB,
								    rhsB,
								    expandedDomainCellLabels,
								    dx,
								    false /* smooth even tiles */,
								    false /* iterate forward */);

		interiorTiledGaussSeidelPoissonSmoother<SolveReal>(solutionB,
								    rhsB,
								    expandedDomainCellLabels,
								    dx,
								    true /* smooth odd tiles */,
								    false /* iterate forward */);
		
		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    expandedDomainCellLabels,
							    boundaryCells,
							    dx,
							    &dummyWeights);							    
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Weighted Gauss Seidel smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    exint interiorCellCount = 0;
	    UT_VoxelArray<exint> directSolverIndices;
	    directSolverIndices.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    directSolverIndices.constant(-1);

	    forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	    {
		if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL||
		    expandedDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    directSolverIndices.setValue(cell, interiorCellCount++);
	    });

	    // Build rows
	    std::vector<Eigen::Triplet<SolveReal>> sparseElements;

	    fpreal32 gridScale = 1. / (dx * dx);
	    forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	    {
		if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
		    expandedDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		{
		    fpreal32 diagonal = 0;
		    int index = directSolverIndices(cell);
		    assert(index >= 0);
		    for (int axis : {0, 1, 2})
			for (int direction : {0, 1})
			{
			    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

			    auto adjacentCellLabels = expandedDomainCellLabels(adjacentCell);
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

	    // Solve system
	    Eigen::SparseMatrix<SolveReal> sparseMatrix(interiorCellCount, interiorCellCount);
	    sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
	    sparseMatrix.makeCompressed();

	    Eigen::SimplicialCholesky<Eigen::SparseMatrix<SolveReal>> solver(sparseMatrix);
	    assert(solver.info() == Eigen::Success);

	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionA.constant(0);
	    
	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionB.constant(0);

	    using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

	    {
		Vector rhsVector = Vector::Zero(interiorCellCount);
		
		forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
		{
		    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			expandedDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			rhsVector(index) = rhsA(cell);
		    }
		});

		Vector solutionVector = solver.solve(rhsVector);

		forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
		{
		    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			expandedDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			solutionA.setValue(cell, solutionVector(index));
		    }
		});
	    }

	    {
		Vector rhsVector = Vector::Zero(interiorCellCount);
		
		forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
		{
		    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			expandedDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			rhsVector(index) = rhsB(cell);
		    }
		});

		Vector solutionVector = solver.solve(rhsVector);

		forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
		{
		    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
			expandedDomainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			solutionB.setValue(cell, solutionVector(index));
		    }
		});
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Direct solve symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    // Test down and up sampling
	    UT_VoxelArray<int> coarseDomainLabels = buildCoarseCellLabels(expandedDomainCellLabels);
	    
	    assert(unitTestBoundaryCells(coarseDomainLabels) && unitTestBoundaryCells(expandedDomainCellLabels));
	    assert(unitTestCoarsening(coarseDomainLabels, expandedDomainCellLabels));

	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionA.constant(0);
	    
	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionB.constant(0);

	    UT_Vector3I coarseGridSize = coarseDomainLabels.getVoxelRes();
		
	    {
		UT_VoxelArray<StoreReal> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseRhs.constant(0);

		downsample<SolveReal>(coarseRhs, rhsA, coarseDomainLabels, expandedDomainCellLabels);
		upsampleAndAdd<SolveReal>(solutionA, coarseRhs, expandedDomainCellLabels, coarseDomainLabels);
	    }
	    {
		UT_VoxelArray<StoreReal> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseRhs.constant(0);

		downsample<SolveReal>(coarseRhs, rhsB, coarseDomainLabels, expandedDomainCellLabels);
		upsampleAndAdd<SolveReal>(solutionB, coarseRhs, expandedDomainCellLabels, coarseDomainLabels);
	    }		

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Restriction/prolongation symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    using Vector = std::conditional<std::is_same<SolveReal, float>::value, Eigen::VectorXf, Eigen::VectorXd>::type;

	    // Test single level correction
	    UT_VoxelArray<int> coarseDomainLabels = buildCoarseCellLabels(expandedDomainCellLabels);

	    assert(unitTestBoundaryCells(coarseDomainLabels) && unitTestBoundaryCells(expandedDomainCellLabels));
	    assert(unitTestCoarsening(coarseDomainLabels, expandedDomainCellLabels));

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

	    fpreal32 gridScale = 1. / (dx * dx);
	    forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
	    {
		if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL ||
		    coarseDomainLabels(cell) == CellLabels::BOUNDARY_CELL)
		{
		    fpreal32 diagonal = 0;
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
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionA.constant(0);
	    
	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionB.constant(0);

	    UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(expandedDomainCellLabels, 2);

	    // Uncompress tiles on the boundary
	    uncompressBoundaryTiles(solutionA, boundaryCells);
	    uncompressBoundaryTiles(solutionB, boundaryCells);

	    {
		interiorJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    expandedDomainCellLabels,
							    dx);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    expandedDomainCellLabels,
							    boundaryCells,
							    dx,
							    &dummyWeights);

		// Compute new residual
		UT_VoxelArray<StoreReal> residual;
		residual.size(expandedGridSize, expandedGridSize, expandedGridSize);
		residual.constant(0);

		computePoissonResidual<SolveReal>(residual, solutionA, rhsA, expandedDomainCellLabels, dx, &dummyWeights);
		
		UT_VoxelArray<StoreReal> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseRhs.constant(0);
		
		downsample<SolveReal>(coarseRhs, residual, coarseDomainLabels, expandedDomainCellLabels);

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

		upsampleAndAdd<SolveReal>(solutionA, coarseSolution, expandedDomainCellLabels, coarseDomainLabels);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    expandedDomainCellLabels,
							    boundaryCells,
							    dx,
							    &dummyWeights);

		interiorJacobiPoissonSmoother<SolveReal>(solutionA,
							    rhsA,
							    expandedDomainCellLabels,
							    dx);
	    }
	    {
		interiorJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    expandedDomainCellLabels,
							    dx);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    expandedDomainCellLabels,
							    boundaryCells,
							    dx,
							    &dummyWeights);

		// Compute new residual
		UT_VoxelArray<StoreReal> residual;
		residual.size(expandedGridSize, expandedGridSize, expandedGridSize);
		residual.constant(0);

		computePoissonResidual<SolveReal>(residual, solutionB, rhsB, expandedDomainCellLabels, dx, &dummyWeights);
		
		UT_VoxelArray<StoreReal> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		coarseRhs.constant(0);
		
		downsample<SolveReal>(coarseRhs, residual, coarseDomainLabels, expandedDomainCellLabels);

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

		upsampleAndAdd<SolveReal>(solutionB, coarseSolution, expandedDomainCellLabels, coarseDomainLabels);

		boundaryJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    expandedDomainCellLabels,
							    boundaryCells,
							    dx,
							    &dummyWeights);

		interiorJacobiPoissonSmoother<SolveReal>(solutionB,
							    rhsB,
							    expandedDomainCellLabels,
							    dx);
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  One level v cycle symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionA.constant(0);

	    {
		HDK::GeometricMultigridPoissonSolver mgSolver(expandedDomainCellLabels, 4, dx, 2, 1, false);
		mgSolver.setBoundaryWeights(dummyWeights);

		mgSolver.applyVCycle(solutionA, rhsA);
		mgSolver.applyVCycle(solutionA, rhsA, true);
		mgSolver.applyVCycle(solutionA, rhsA, true);
		mgSolver.applyVCycle(solutionA, rhsA, true);
	    }

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionB.constant(0);

	    {
		HDK::GeometricMultigridPoissonSolver mgSolver(expandedDomainCellLabels, 4, dx, 2, 1, false);
		mgSolver.setBoundaryWeights(dummyWeights);

		mgSolver.applyVCycle(solutionB, rhsB);
		mgSolver.applyVCycle(solutionB, rhsB, true);
		mgSolver.applyVCycle(solutionB, rhsB, true);
		mgSolver.applyVCycle(solutionB, rhsB, true);
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Test multigrid poisson solver symmetry with Jacobi smoothing. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
    	{
	    UT_VoxelArray<StoreReal> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionA.constant(0);

	    {
		HDK::GeometricMultigridPoissonSolver mgSolver(expandedDomainCellLabels, 4, dx, 2, 1, true);
		mgSolver.setBoundaryWeights(dummyWeights);

		mgSolver.applyVCycle(solutionA, rhsA);
		mgSolver.applyVCycle(solutionA, rhsA, true);
		mgSolver.applyVCycle(solutionA, rhsA, true);
		mgSolver.applyVCycle(solutionA, rhsA, true);
	    }

	    UT_VoxelArray<StoreReal> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    solutionB.constant(0);

	    {
		HDK::GeometricMultigridPoissonSolver mgSolver(expandedDomainCellLabels, 4, dx, 2, 1, true);
		mgSolver.setBoundaryWeights(dummyWeights);

		mgSolver.applyVCycle(solutionB, rhsB);
		mgSolver.applyVCycle(solutionB, rhsB, true);
		mgSolver.applyVCycle(solutionB, rhsB, true);
		mgSolver.applyVCycle(solutionB, rhsB, true);
	    }

	    fpreal32 dotA = dotProduct<SolveReal>(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct<SolveReal>(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Test multigrid poisson solver symmetry with Gauss Seidel smoothing. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
    }
    if (getTestOneLevelVCycle())
    {
	std::cout << "\n// Testing one level v-cycle" << std::endl;

	UT_VoxelArray<StoreReal> solutionGrid;
	solutionGrid.size(gridSize, gridSize, gridSize);
	solutionGrid.constant(0);

	UT_Interrupt *boss = UTgetInterrupt();

	{
	    std::cout << "  Build initial guess" << std::endl;

	    // We have implicity built a grid sized [0,1]x[0,1]x[0,1]
	    // with a dx of 1/gridSize

	    fpreal dx = 1. / fpreal(gridSize);

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
	rhsGrid.size(gridSize, gridSize, gridSize);
	rhsGrid.constant(0);

	UT_VoxelArray<StoreReal> residualGrid;
	residualGrid.size(gridSize, gridSize, gridSize);
	residualGrid.constant(0);

	computePoissonResidual<SolveReal>(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

	fpreal infNormError = infNorm(residualGrid, domainCellLabels);
	fpreal l2NormError = l2Norm<SolveReal>(residualGrid, domainCellLabels);

	std::cout << "L-infinity norm: " << infNormError << std::endl;
	std::cout << "L-2 norm: " << l2NormError << std::endl;

	std::cout << "  Build dummy gradient weights" << std::endl;

	// Build dummy weights
	std::array<UT_VoxelArray<StoreReal>, 3> dummyWeights;
	for (int axis : {0,1,2})
	{
	    UT_Vector3I size(gridSize, gridSize, gridSize);
	    ++size[axis];
	    dummyWeights[axis].size(size[0], size[1], size[2]);
	    dummyWeights[axis].constant(0);

	    UTparallelForEachNumber(dummyWeights[axis].numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<StoreReal> vit;
		vit.setConstArray(&dummyWeights[axis]);
		UT_VoxelTileIterator<StoreReal> vitt;

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
			    UT_Vector3I face(vitt.x(), vitt.y(), vitt.z());
			
			    bool isInterior = false;
			    bool isExterior = false;
			    for (int direction : {0,1})
			    {
				UT_Vector3I cell = faceToCellMap(face, axis, direction);

				if (cell[axis] < 0 || cell[axis] >= domainCellLabels.getVoxelRes()[axis])
				    continue;
				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL ||
				    domainCellLabels(cell) == CellLabels::BOUNDARY_CELL)
				    isInterior = true;
				else if (domainCellLabels(cell) == CellLabels::EXTERIOR_CELL)
				    isExterior = true;
			    }

			    if (isInterior && !isExterior)
				dummyWeights[axis].setValue(face, 1);
			}
		    }
		}
	    });
	}

	std::cout << "  Apply one level of v-cycle" << std::endl;

	// Pre-build multigrid preconditioner
	HDK::GeometricMultigridPoissonSolver mgSolver(domainCellLabels, 2 /* levels in v-cycle */, dx, 3 /* boundary width */, 1, false);
	mgSolver.setBoundaryWeights(dummyWeights);

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
	rhsGrid.size(gridSize, gridSize, gridSize);
	rhsGrid.constant(0);

	UT_VoxelArray<StoreReal> solutionGrid;
	solutionGrid.size(gridSize, gridSize, gridSize);
	solutionGrid.constant(0);

	UT_VoxelArray<StoreReal> residualGrid;
	residualGrid.size(gridSize, gridSize, gridSize);
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
	fpreal32 deltaPercent = .1;
	UT_Vector3I deltaPoint = UT_Vector3I(deltaPercent * UT_Vector3(gridSize));

	{
	    fpreal32 deltaAmplitude = getDeltaFunctionAmplitude();

	    const UT_Vector3I startCell = deltaPoint - UT_Vector3I(1);
	    const UT_Vector3I endCell = deltaPoint + UT_Vector3I(2);
	    UT_Vector3I sampleCell;

	    for (sampleCell[0] = startCell[0]; sampleCell[0] < endCell[0]; ++sampleCell[0])
		for (sampleCell[1] = startCell[1]; sampleCell[1] < endCell[1]; ++sampleCell[1])
		    for (sampleCell[2] = startCell[2]; sampleCell[2] < endCell[2]; ++sampleCell[2])
			rhsGrid.setValue(sampleCell, deltaAmplitude);
	}

	UT_Array<UT_Vector3I> boundaryCells = buildBoundaryCells(domainCellLabels, 2);

	using namespace HDK::GeometricMultigridOperators;

	// Uncompress tiles on the boundary
	uncompressBoundaryTiles(solutionGrid, boundaryCells);

	// Apply smoother
	const int maxSmootherIterations = getMaxSmootherIterations();
   
	const bool useGaussSeidelSmoothing = getUseGaussSeidelSmoothing();

	int iteration = 0;
	fpreal64 solvetime = 0;
	fpreal64 boundarysolvetime = 0;
	fpreal64 count = 0;
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
		boundaryJacobiPoissonSmoother<SolveReal>(solutionGrid, rhsGrid, domainCellLabels, boundaryCells, dx);

		boundarysolvetime += timer.stop();
		timer.clear();
		timer.start();
	    }

	    computePoissonResidual<SolveReal>(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

	    fpreal infNormError = infNorm(residualGrid, domainCellLabels);
	    fpreal l2NormError = l2Norm<SolveReal>(residualGrid, domainCellLabels);

	    std::cout << "Iteration: " << iteration << std::endl;
	    std::cout << "L-infinity norm: " << infNormError << std::endl;
	    std::cout << "L-2 norm: " << l2NormError << std::endl;
	}

	std::cout << "Interior smoother time: " << solvetime / count << std::endl;
	std::cout << "Boundary smoother time: " << boundarysolvetime / count << std::endl;
    }

    return true;
}