#include "HDK_TestGeometricMultiGrid.h"

#include <PRM/PRM_Include.h>

#include <SIM/SIM_DopDescription.h>
#include <SIM/SIM_FieldUtils.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_PRMShared.h>

#include <UT/UT_DSOVersion.h>
#include <UT/UT_ParallelUtil.h>
#include <UT/UT_PerfMonAutoEvent.h>

#include "HDK_GeometricMultiGridOperations.h"
#include "HDK_GeometricMultiGridPoissonSolver.h"

using namespace HDK::GeometricMultiGridOperations;
using namespace SIM::FieldUtils;

void initializeSIM(void *)
{
   IMPLEMENT_DATAFACTORY(HDK_TestGeometricMultiGrid);
}

// Standard constructor, note that BaseClass was crated by the
// DECLARE_DATAFACTORY and provides an easy way to chain through
// the class hierarchy.
HDK_TestGeometricMultiGrid::HDK_TestGeometricMultiGrid(const SIM_DataFactory *factory)
    : BaseClass(factory)
{
}

HDK_TestGeometricMultiGrid::~HDK_TestGeometricMultiGrid()
{
}

const SIM_DopDescription* HDK_TestGeometricMultiGrid::getDopDescription()
{
    static PRM_Name	theGridSizeName("gridSize", "Grid Size");
    static PRM_Default  theGridSizeDefault(64);

    static PRM_Name	theUseComplexDomainName("useComplexDomain", "Use Complex Domain");

    static PRM_Name	theUseSolidSphereName("useSolidSphere", "Use Solid Sphere");
    static PRM_Conditional    theUseSolidSphereDisable("{ useComplexDomain == 0 }");

    static PRM_Name theTestSymmetrySeparatorName("testSymmetrySeparator", "Test Symmetry Separator");

    // Test symmetry

    static PRM_Name	theTestSymmetryName("testSymmetry", "Test Symmetry");


    static PRM_Name theTestOneLevelVCycleSeparatorName("testOneLevelVCycleSeparator", "One Level V-Cycle Separator");

    // Test one level v-cycle parameters

    static PRM_Name	theTestOneLevelVCycleName("testOneLevelVCycle", "Test One Level V-cycle");

    // Test smoother parameters

    static PRM_Name theSmootherSeparatorName("smootherSeparator", "Smoother Separator");

    static PRM_Name	theTestSmootherName("testSmoother", "Test Smoother");
    static PRM_Conditional    theSmootherParameterDisable("{ testSmoother == 0 }");

    static PRM_Name	theUseRandomInitialGuessName("useRandomInitialGuess", "Use Random Initial Guess");

    static PRM_Name	theDeltaFunctionAmplitudeName("deltaFunctionAmplitude", "Delta Function Amplitude");
    static PRM_Default	theDeltaFunctionAmplitudeDefault(1000);

    static PRM_Name	theMaxSmootherIterationsName("maxSmootherIterations", "Max Smoother Iterations");
    static PRM_Default	theMaxSmootherIterationsDefault(1000);

    static PRM_Name	theUseGaussSeidelSmoothingName("useGaussSeidelSmoothing", "Use Gauss Seidel Smoothing");

    static PRM_Template	theTemplates[] =
    {
	PRM_Template(PRM_INT, 1, &theGridSizeName, &theGridSizeDefault),

	PRM_Template(PRM_TOGGLE, 1, &theUseComplexDomainName),

	PRM_Template(PRM_TOGGLE, 1, &theUseSolidSphereName, PRMzeroDefaults,
			0, 0, 0, 0, 1, 0, &theUseSolidSphereDisable),

	// Symmetry test parameters
	PRM_Template(PRM_SEPARATOR, 1, &theTestSymmetrySeparatorName),

	PRM_Template(PRM_TOGGLE, 1, &theTestSymmetryName),

	// One level v-cycle parameters
	PRM_Template(PRM_SEPARATOR, 1, &theTestOneLevelVCycleSeparatorName),

	PRM_Template(PRM_TOGGLE, 1, &theTestOneLevelVCycleName),


	// Smoother test parameters
	PRM_Template(PRM_SEPARATOR, 1, &theSmootherSeparatorName),

	PRM_Template(PRM_TOGGLE, 1, &theTestSmootherName),

	PRM_Template(PRM_TOGGLE, 1, &theUseRandomInitialGuessName, PRMzeroDefaults,
			0, 0, 0, 0, 1, 0, &theSmootherParameterDisable),

	PRM_Template(PRM_FLT, 1, &theDeltaFunctionAmplitudeName, &theDeltaFunctionAmplitudeDefault,
			0, 0, 0, 0, 1, 0, &theSmootherParameterDisable),

	PRM_Template(PRM_INT, 1, &theMaxSmootherIterationsName, &theMaxSmootherIterationsDefault,
			0, 0, 0, 0, 1, 0, &theSmootherParameterDisable),

	PRM_Template(PRM_TOGGLE, 1, &theUseGaussSeidelSmoothingName, PRMzeroDefaults,
			0, 0, 0, 0, 1, 0, &theSmootherParameterDisable),

	    PRM_Template()
    };

    static SIM_DopDescription theDopDescription(true,
						"HDK_TestGeometricMultiGrid",
						"HDK Test Geometric Multi Grid",
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

    return domainCellLabels;
}

bool HDK_TestGeometricMultiGrid::solveGasSubclass(SIM_Engine &engine,
						    SIM_Object *obj,
						    SIM_Time time,
						    SIM_Time timestep)
{
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

	UT_VoxelArray<fpreal32> rhsA;
	rhsA.size(expandedGridSize, expandedGridSize, expandedGridSize);

	UT_VoxelArray<fpreal32> rhsB;
	rhsB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	{
	    std::default_random_engine generator;
	    std::uniform_real_distribution<fpreal32> distribution(0, 1);

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
	UT_VoxelArray<fpreal32> dummyWeights[3];
	for (int axis : {0,1,2})
	{
	    UT_Vector3I size(expandedGridSize, expandedGridSize, expandedGridSize);
	    ++size[axis];
	    dummyWeights[axis].size(size[0], size[1], size[2]);

	    forEachVoxelRange(UT_Vector3I(exint(0)), dummyWeights[axis].getVoxelRes(), [&](const UT_Vector3I &face)
	    {
		bool isInterior = false;
		bool isExterior = false;
		for (int direction : {0, 1})
		{
		    UT_Vector3I cell = faceToCellMap(face, axis, direction);

		    if (cell[axis] < 0 || cell[axis] >= expandedDomainCellLabels.getVoxelRes()[axis])
			continue;

		    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
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
	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    // Test Jacobi symmetry
	    dampedJacobiPoissonSmoother(solutionA,
					rhsA,
					expandedDomainCellLabels,
					dummyWeights,
					dx);

	    dampedJacobiPoissonSmoother(solutionB,
					rhsB,
					expandedDomainCellLabels,
					dummyWeights,
					dx);


	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Weighted Jacobi smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    // Test Jacobi symmetry
	    dampedJacobiPoissonSmoother(solutionA,
					rhsA,
					expandedDomainCellLabels,
					dx);

	    dampedJacobiPoissonSmoother(solutionB,
					rhsB,
					expandedDomainCellLabels,
					dx);


	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);
	    
	    std::cout << "  Jacobi smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    UT_Array<UT_Vector3I> tempInteriorCells;

	    forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	    {
		if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
		{
		    tempInteriorCells.append(cell);
		}
	    });

	    // Uncompress tiles for solution grids
	    for (exint i = 0; i < tempInteriorCells.size(); ++i)
	    {
		UT_Vector3I cell = tempInteriorCells[i];
		int tileNumber = solutionA.indexToLinearTile(cell[0], cell[1], cell[2]);
		if (solutionA.getLinearTile(tileNumber)->isConstant())
		    solutionA.getLinearTile(tileNumber)->uncompress();

		assert(tileNumber == solutionB.indexToLinearTile(cell[0], cell[1], cell[2]));
		if (solutionB.getLinearTile(tileNumber)->isConstant())
		    solutionB.getLinearTile(tileNumber)->uncompress();
	    }

	    dampedJacobiPoissonSmoother(solutionA,
					rhsA,
					expandedDomainCellLabels,
					tempInteriorCells,
					dummyWeights,
					dx);

	    dampedJacobiPoissonSmoother(solutionB,
					rhsB,
					expandedDomainCellLabels,
					tempInteriorCells,
					dummyWeights,
					dx);

	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Weighted boundary Jacobi smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    UT_Array<UT_Vector3I> tempInteriorCells;

	    forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	    {
		if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
		{
		    tempInteriorCells.append(cell);
		}
	    });

	    // Uncompress tiles for solution grids
	    for (exint i = 0; i < tempInteriorCells.size(); ++i)
	    {
		UT_Vector3I cell = tempInteriorCells[i];
		int tileNumber = solutionA.indexToLinearTile(cell[0], cell[1], cell[2]);
		if (solutionA.getLinearTile(tileNumber)->isConstant())
		    solutionA.getLinearTile(tileNumber)->uncompress();

		assert(tileNumber == solutionB.indexToLinearTile(cell[0], cell[1], cell[2]));
		if (solutionB.getLinearTile(tileNumber)->isConstant())
		    solutionB.getLinearTile(tileNumber)->uncompress();
	    }

	    dampedJacobiPoissonSmoother(solutionA,
					rhsA,
					expandedDomainCellLabels,
					tempInteriorCells,
					dx);

	    dampedJacobiPoissonSmoother(solutionB,
					rhsB,
					expandedDomainCellLabels,
					tempInteriorCells,
					dx);

	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Boundary Jacobi smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	// Test Gauss Seidel smoother
	{
	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    // Test Jacobi symmetry
	    for (int iteration = 0; iteration < 4; ++iteration)
	    {
		tiledGaussSeidelPoissonSmoother(solutionA,
						rhsA,
						expandedDomainCellLabels,
						dummyWeights,
						dx,
						true /* smooth odd tiles */,
						true /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionA,
						rhsA,
						expandedDomainCellLabels,
						dummyWeights,
						dx,
						false /* smooth odd tiles */,
						true /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionA,
						rhsA,
						expandedDomainCellLabels,
						dummyWeights,
						dx,
						false /* smooth odd tiles */,
						false /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionA,
						rhsA,
						expandedDomainCellLabels,
						dummyWeights,
						dx,
						true /* smooth odd tiles */,
						false /* iterate forward */);
	    }

	    for (int iteration = 0; iteration < 4; ++iteration)
	    {
		tiledGaussSeidelPoissonSmoother(solutionB,
						rhsB,
						expandedDomainCellLabels,
						dummyWeights,
						dx,
						true /* smooth odd tiles */,
						true /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionB,
						rhsB,
						expandedDomainCellLabels,
						dummyWeights,
						dx,
						false /* smooth odd tiles */,
						true /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionB,
						rhsB,
						expandedDomainCellLabels,
						dummyWeights,
						dx,
						false /* smooth odd tiles */,
						false /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionB,
						rhsB,
						expandedDomainCellLabels,
						dummyWeights,
						dx,
						true /* smooth odd tiles */,
						false /* iterate forward */);
	    }



	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Weighted Gauss Seidel smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    for (int iteration = 0; iteration < 4; ++iteration)
	    {
		tiledGaussSeidelPoissonSmoother(solutionA,
						rhsA,
						expandedDomainCellLabels,
						dx,
						true /* smooth odd tiles */,
						true /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionA,
						rhsA,
						expandedDomainCellLabels,
						dx,
						false /* smooth odd tiles */,
						true /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionA,
						rhsA,
						expandedDomainCellLabels,
						dx,
						false /* smooth odd tiles */,
						false /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionA,
						rhsA,
						expandedDomainCellLabels,
						dx,
						true /* smooth odd tiles */,
						false /* iterate forward */);
	    }

	    for (int iteration = 0; iteration < 4; ++iteration)
	    {
		tiledGaussSeidelPoissonSmoother(solutionB,
						rhsB,
						expandedDomainCellLabels,
						dx,
						true /* smooth odd tiles */,
						true /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionB,
						rhsB,
						expandedDomainCellLabels,
						dx,
						false /* smooth odd tiles */,
						true /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionB,
						rhsB,
						expandedDomainCellLabels,
						dx,
						false /* smooth odd tiles */,
						false /* iterate forward */);

		tiledGaussSeidelPoissonSmoother(solutionB,
						rhsB,
						expandedDomainCellLabels,
						dx,
						true /* smooth odd tiles */,
						false /* iterate forward */);
	    }


	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Gauss Seidel smoothing symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}


	// {
	//     // Test direct solve symmetry
	//     Eigen::SimplicialCholesky<Eigen::SparseMatrix<fpreal32>> solver;
	//     Eigen::SparseMatrix<fpreal32> sparseMatrix;

	//     exint interiorCellCount = 0;
	//     UT_VoxelArray<exint> directSolverIndices;
	//     directSolverIndices.size(expandedGridSize, expandedGridSize, expandedGridSize);
	//     directSolverIndices.constant(-1);

	//     forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	//     {
	// 	if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	// 	    directSolverIndices.setValue(cell, interiorCellCount++);
	//     });

	//     // Build rows
	//     std::vector<Eigen::Triplet<fpreal32>> sparseElements;

	//     fpreal32 gridScale = 1. / (dx * dx);
	//     forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	//     {
	// 	if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	// 	{
	// 	    fpreal32 diagonal = 0;
	// 	    int index = directSolverIndices(cell);
	// 	    assert(index >= 0);
	// 	    for (int axis : {0, 1, 2})
	// 		for (int direction : {0, 1})
	// 		{
	// 		    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

	// 		    auto cellLabels = expandedDomainCellLabels(adjacentCell);
	// 		    if (cellLabels == CellLabels::INTERIOR_CELL)
	// 		    {
	// 			exint adjacentIndex = directSolverIndices(adjacentCell);
	// 			assert(adjacentIndex >= 0);

	// 			sparseElements.push_back(Eigen::Triplet<fpreal32>(index, adjacentIndex, -gridScale));
	// 			diagonal += gridScale;
	// 		    }
	// 		    else if (cellLabels == CellLabels::DIRICHLET_CELL)
	// 			diagonal += gridScale;
	// 		}

	// 	    sparseElements.push_back(Eigen::Triplet<fpreal32>(index, index, diagonal));
	// 	}
	//     });

	//     // Solve system
	//     sparseMatrix = Eigen::SparseMatrix<fpreal32>(interiorCellCount, interiorCellCount);
	//     sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
	//     sparseMatrix.makeCompressed();

	//     solver.compute(sparseMatrix);

	//     assert(solver.info() == Eigen::Success);

	//     UT_VoxelArray<fpreal32> solutionA;
	//     solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	//     UT_VoxelArray<fpreal32> solutionB;
	//     solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	//     {
	// 	Eigen::VectorXf rhsVector = Eigen::VectorXf::Zero(interiorCellCount);
		
	// 	forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	// 	{
	// 	    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	// 	    {
	// 		exint index = directSolverIndices(cell);
	// 		assert(index >= 0);

	// 		rhsVector(index) = rhsA(cell);
	// 	    }
	// 	});

	// 	Eigen::VectorXf solutionVector = solver.solve(rhsVector);

	// 	forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	// 	{
	// 	    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	// 	    {
	// 		exint index = directSolverIndices(cell);
	// 		assert(index >= 0);

	// 		solutionA.setValue(cell, solutionVector(index));
	// 	    }
	// 	});
	//     }

	//     {
	// 	Eigen::VectorXf rhsVector = Eigen::VectorXf::Zero(interiorCellCount);
		
	// 	forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	// 	{
	// 	    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	// 	    {
	// 		exint index = directSolverIndices(cell);
	// 		assert(index >= 0);

	// 		rhsVector(index) = rhsB(cell);
	// 	    }
	// 	});

	// 	Eigen::VectorXf solutionVector = solver.solve(rhsVector);

	// 	forEachVoxelRange(UT_Vector3I(exint(0)), UT_Vector3I(exint(expandedGridSize)), [&](const UT_Vector3I &cell)
	// 	{
	// 	    if (expandedDomainCellLabels(cell) == CellLabels::INTERIOR_CELL)
	// 	    {
	// 		exint index = directSolverIndices(cell);
	// 		assert(index >= 0);

	// 		solutionB.setValue(cell, solutionVector(index));
	// 	    }
	// 	});
	//     }

	//     fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	//     fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	//     std::cout << "  Direct solve symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	//     assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	// }
	
	{
		// Test down and up sampling
	    UT_VoxelArray<int> coarseDomainLabels = buildCoarseCellLabels(expandedDomainCellLabels);
	    
	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    UT_Vector3I coarseGridSize = coarseDomainLabels.getVoxelRes();
		
	    {
		UT_VoxelArray<fpreal32> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);

		downsample(coarseRhs, rhsA, coarseDomainLabels, expandedDomainCellLabels);
		upsampleAndAdd(solutionA, coarseRhs, expandedDomainCellLabels, coarseDomainLabels);
	    }
	    {
		UT_VoxelArray<fpreal32> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);

		downsample(coarseRhs, rhsB, coarseDomainLabels, expandedDomainCellLabels);
		upsampleAndAdd(solutionB, coarseRhs, expandedDomainCellLabels, coarseDomainLabels);
	    }		

	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Restriction/prolongation symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
	    // Test single level correction
	    UT_VoxelArray<int> coarseDomainLabels = buildCoarseCellLabels(expandedDomainCellLabels);
	    UT_Vector3I coarseGridSize = coarseDomainLabels.getVoxelRes();

	    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver;
	    Eigen::SparseMatrix<double> sparseMatrix;

	    exint interiorCellCount = 0;
	    UT_VoxelArray<exint> directSolverIndices;

	    directSolverIndices.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
	    directSolverIndices.constant(-1);

	    forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
	    {
		if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL)
		    directSolverIndices.setValue(cell, interiorCellCount++);
	    });

	    std::vector<Eigen::Triplet<double>> sparseElements;

	    fpreal32 gridScale = 1. / (dx * dx);
	    forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
	    {
		if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL)
		{
		    fpreal32 diagonal = 0;
		    int index = directSolverIndices(cell);
		    assert(index >= 0);
		    for (int axis : {0, 1, 2})
			for (int direction : {0, 1})
			{
			    UT_Vector3I adjacentCell = cellToCellMap(cell, axis, direction);

			    auto cellLabels = coarseDomainLabels(adjacentCell);
			    if (cellLabels == CellLabels::INTERIOR_CELL)
			    {
				exint adjacentIndex = directSolverIndices(adjacentCell);
				assert(adjacentIndex >= 0);

				sparseElements.push_back(Eigen::Triplet<double>(index, adjacentIndex, -gridScale));
				diagonal += gridScale;
			    }
			    else if (cellLabels == CellLabels::DIRICHLET_CELL)
				diagonal += gridScale;
			}

		    sparseElements.push_back(Eigen::Triplet<double>(index, index, diagonal));
		}
	    });

	    // Solve system
	    sparseMatrix = Eigen::SparseMatrix<double>(interiorCellCount, interiorCellCount);
	    sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());
	    sparseMatrix.makeCompressed();

	    solver.compute(sparseMatrix);

	    assert(solver.info() == Eigen::Success);

	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);
	    
	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    {
		// Pre-smooth to get an initial guess
		dampedJacobiPoissonSmoother(solutionA, rhsA, expandedDomainCellLabels, dummyWeights, dx);

		// Compute new residual
		UT_VoxelArray<fpreal32> residual;
		residual.size(expandedGridSize, expandedGridSize, expandedGridSize);

		computePoissonResidual(residual, solutionA, rhsA, expandedDomainCellLabels, dummyWeights, dx);
		
		UT_VoxelArray<fpreal32> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		
		downsample(coarseRhs, residual, coarseDomainLabels, expandedDomainCellLabels);

		Eigen::VectorXd rhsVector = Eigen::VectorXd::Zero(interiorCellCount);
		
		// Copy to Eigen and direct solve
		forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
		{
		    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			rhsVector(index) = coarseRhs(cell);
		    }
		});

		UT_VoxelArray<fpreal32> coarseSolution;
		coarseSolution.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);

		Eigen::VectorXd solution = solver.solve(rhsVector);

		// Copy solution back
		forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
		{
		    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			coarseSolution.setValue(cell, solution(index));
		    }
		});

		upsampleAndAdd(solutionA, coarseSolution, expandedDomainCellLabels, coarseDomainLabels);

		dampedJacobiPoissonSmoother(solutionA, rhsA, expandedDomainCellLabels, dummyWeights, dx);
	    }
	    {
		// Pre-smooth to get an initial guess
		dampedJacobiPoissonSmoother(solutionB, rhsB, expandedDomainCellLabels, dummyWeights, dx);

		// Compute new residual
		UT_VoxelArray<fpreal32> residual;
		residual.size(expandedGridSize, expandedGridSize, expandedGridSize);

		computePoissonResidual(residual, solutionB, rhsB, expandedDomainCellLabels, dummyWeights, dx);
		
		UT_VoxelArray<fpreal32> coarseRhs;
		coarseRhs.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);
		
		downsample(coarseRhs, residual, coarseDomainLabels, expandedDomainCellLabels);

		Eigen::VectorXd rhsVector = Eigen::VectorXd::Zero(interiorCellCount);
		
		// Copy to Eigen and direct solve
		forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
		{
		    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			rhsVector(index) = coarseRhs(cell);
		    }
		});

		UT_VoxelArray<fpreal32> coarseSolution;
		coarseSolution.size(coarseGridSize[0], coarseGridSize[1], coarseGridSize[2]);

		Eigen::VectorXd solution = solver.solve(rhsVector);

		// Copy solution back
		forEachVoxelRange(UT_Vector3I(exint(0)), coarseGridSize, [&](const UT_Vector3I &cell)
		{
		    if (coarseDomainLabels(cell) == CellLabels::INTERIOR_CELL)
		    {
			exint index = directSolverIndices(cell);
			assert(index >= 0);

			coarseSolution.setValue(cell, solution(index));
		    }
		});

		upsampleAndAdd(solutionB, coarseSolution, expandedDomainCellLabels, coarseDomainLabels);

		dampedJacobiPoissonSmoother(solutionB, rhsB, expandedDomainCellLabels, dummyWeights, dx);
	    }

	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  One level v cycle symmetry test. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}

	{
	    UT_VoxelArray<fpreal32> solutionA;
	    solutionA.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    {
		HDK::GeometricMultiGridPoissonSolver mgSolver(expandedDomainCellLabels, 4, dx, 2, 1, 1, true);
		mgSolver.setGradientWeights(dummyWeights);

		mgSolver.applyVCycle(solutionA, rhsA);
		// mgSolver.applyVCycle(solutionA, rhsA, true);
		// mgSolver.applyVCycle(solutionA, rhsA, true);
		// mgSolver.applyVCycle(solutionA, rhsA, true);
	    }

	    UT_VoxelArray<fpreal32> solutionB;
	    solutionB.size(expandedGridSize, expandedGridSize, expandedGridSize);

	    {
		HDK::GeometricMultiGridPoissonSolver mgSolver(expandedDomainCellLabels, 4, dx, 2, 1, 1, true);
		mgSolver.setGradientWeights(dummyWeights);

		mgSolver.applyVCycle(solutionB, rhsB);
		// mgSolver.applyVCycle(solutionB, rhsB, true);
		// mgSolver.applyVCycle(solutionB, rhsB, true);
		// mgSolver.applyVCycle(solutionB, rhsB, true);
	    }

	    fpreal32 dotA = dotProduct(solutionA, rhsB, expandedDomainCellLabels);
	    fpreal32 dotB = dotProduct(solutionB, rhsA, expandedDomainCellLabels);

	    std::cout << "  Test multigrid poisson solver symmetry. BMA: " << dotB << ". AMB: " << dotA << std::endl;

	    assert(fabs(dotA - dotB) / fabs(std::max(dotA, dotB)) < 1E-10);
	}
    }

    if (getTestOneLevelVCycle())
    {
	std::cout << "\n// Testing one level v-cycle" << std::endl;

	UT_VoxelArray<fpreal32> solutionGrid;
	solutionGrid.size(gridSize, gridSize, gridSize);

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
			if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
			{
			    vitt.setTile(vit);

			    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			    {
				if (vitt.getValue() == CellLabels::INTERIOR_CELL)
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
	UT_VoxelArray<fpreal32> rhsGrid;
	rhsGrid.size(gridSize, gridSize, gridSize);

	UT_VoxelArray<fpreal32> residualGrid;
	residualGrid.size(gridSize, gridSize, gridSize);

	computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

	fpreal infNormError = infNorm(residualGrid, domainCellLabels);
	fpreal l2NormError = l2Norm(residualGrid, domainCellLabels);

	std::cout << "L-infinity norm: " << infNormError << std::endl;
	std::cout << "L-2 norm: " << l2NormError << std::endl;

	std::cout << "  Build dummy gradient weights" << std::endl;

	// Build dummy weights
	UT_VoxelArray<fpreal32> dummyWeights[3];
	for (int axis : {0,1,2})
	{
	    UT_Vector3I size(gridSize, gridSize, gridSize);
	    ++size[axis];
	    dummyWeights[axis].size(size[0], size[1], size[2]);

	    UTparallelForEachNumber(dummyWeights[axis].numTiles(), [&](const UT_BlockedRange<int> &range)
	    {
		UT_VoxelArrayIterator<fpreal32> vit;
		vit.setConstArray(&dummyWeights[axis]);
		UT_VoxelTileIterator<fpreal32> vitt;

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
				if (domainCellLabels(cell) == CellLabels::INTERIOR_CELL)
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
	HDK::GeometricMultiGridPoissonSolver mgSolver(domainCellLabels, 2 /* levels in v-cycle */, dx, 3 /* boundary width */);
	mgSolver.setGradientWeights(dummyWeights);

	mgSolver.applyVCycle(solutionGrid, rhsGrid, true /* use initial guess */);

	std::cout << "  Compute test residual" << std::endl;

	computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

	infNormError = infNorm(residualGrid, domainCellLabels);
	l2NormError = l2Norm(residualGrid, domainCellLabels);

	std::cout << "L-infinity norm: " << infNormError << std::endl;
	std::cout << "L-2 norm: " << l2NormError << std::endl;
    }

    if (getTestSmoother())
    {
	std::cout << "\n// Testing smoother" << std::endl;
    
	UT_VoxelArray<fpreal32> rhsGrid;
	rhsGrid.size(gridSize, gridSize, gridSize);

	UT_VoxelArray<fpreal32> solutionGrid;
	solutionGrid.size(gridSize, gridSize, gridSize);

	UT_VoxelArray<fpreal32> residualGrid;
	residualGrid.size(gridSize, gridSize, gridSize);

	if (getUseRandomInitialGuess())
	{
	    std::cout << "  Build random initial guess" << std::endl;

	    std::default_random_engine generator;
	    std::uniform_real_distribution<fpreal32> distribution(0, 1);
	
	    UT_Interrupt *boss = UTgetInterrupt();

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
			if (!vit.isTileConstant() || vit.getValue() == CellLabels::INTERIOR_CELL)
			{
			    vitt.setTile(vit);

			    for (vitt.rewind(); !vitt.atEnd(); vitt.advance())
			    {
				if (vitt.getValue() == CellLabels::INTERIOR_CELL)
				{
				    UT_Vector3I cell(vitt.x(), vitt.y(), vitt.z());

				    solutionGrid.setValue(cell, distribution(generator));
				}
			    }
			}
		    }
		}
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

	// Apply smoother
	const int maxSmootherIterations = getMaxSmootherIterations();

	using namespace HDK::GeometricMultiGridOperations;
    
	const bool useGaussSeidelSmoothing = getUseGaussSeidelSmoothing();

	int iteration = 0;
	for (; iteration < maxSmootherIterations; ++iteration)
	{
	    if (useGaussSeidelSmoothing)
	    {
		tiledGaussSeidelPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, dx, true /*smooth odd tiles*/, true /*smooth forwards*/);
		tiledGaussSeidelPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, dx, false /*smooth even tiles*/, true /*smooth forwards*/);
		tiledGaussSeidelPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, dx, false /*smooth even tiles*/, false /*smooth backwards*/);
		tiledGaussSeidelPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, dx, true /*smooth odd tiles*/, false /*smooth backwards*/);
	    }
	    else dampedJacobiPoissonSmoother(solutionGrid, rhsGrid, domainCellLabels, dx);
	    computePoissonResidual(residualGrid, solutionGrid, rhsGrid, domainCellLabels, dx);

	    fpreal infNormError = infNorm(residualGrid, domainCellLabels);
	    fpreal l2NormError = l2Norm(residualGrid, domainCellLabels);

	    std::cout << "Iteration: " << iteration << std::endl;
	    std::cout << "L-infinity norm: " << infNormError << std::endl;
	    std::cout << "L-2 norm: " << l2NormError << std::endl;
	}
    }

    return true;
}