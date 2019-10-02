#ifndef HDK_TEST_GEOMETRIC_MULTI_GRID_H
#define HDK_TEST_GEOMETRIC_MULTI_GRID_H

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

class GAS_API HDK_TestGeometricMultiGrid : public GAS_SubSolver
{
    // Common parameters
    GET_DATA_FUNC_I("gridSize", GridSize);
    GET_DATA_FUNC_B("useComplexDomain", UseComplexDomain);
    GET_DATA_FUNC_B("useSolidSphere", UseSolidSphere);
    GET_DATA_FUNC_F("sphereRadius", SphereRadius);

    GET_DATA_FUNC_B("useRandomInitialGuess", UseRandomInitialGuess);
    GET_DATA_FUNC_F("deltaFunctionAmplitude", DeltaFunctionAmplitude);

    // Conjugate gradient test
    GET_DATA_FUNC_B("testConjugateGradient", TestConjugateGradient);
    GET_DATA_FUNC_B("useMultiGridPreconditioner", UseMultiGridPreconditioner);
    GET_DATA_FUNC_B("solveCGGeometrically", SolveCGGeometrically);
    GET_DATA_FUNC_I("multiGridLevels", MultiGridLevels);
    GET_DATA_FUNC_F("solverTolerance", SolverTolerance);
    GET_DATA_FUNC_I("maxSolverIterations", MaxSolverIterations);

    // Symmetry
    GET_DATA_FUNC_B("testSymmetry", TestSymmetry);

    // One-level correction
    GET_DATA_FUNC_B("testOneLevelVCycle", TestOneLevelVCycle);

    // Smoother test parameters
    GET_DATA_FUNC_B("testSmoother", TestSmoother);
    GET_DATA_FUNC_I("maxSmootherIterations", MaxSmootherIterations);
    GET_DATA_FUNC_B("useGaussSeidelSmoothing", UseGaussSeidelSmoothing);

public:

    explicit HDK_TestGeometricMultiGrid(const SIM_DataFactory *factory);
    virtual ~HDK_TestGeometricMultiGrid();

    // The overloaded callback that GAS_SubSolver will invoke to
    // perform our actual computation.  We are giving a single object
    // at a time to work on.
    virtual bool solveGasSubclass(SIM_Engine &engine,
				    SIM_Object *obj,
				    SIM_Time time,
				    SIM_Time timestep);

private:

    /// These macros are necessary to bind our node to the factory and
    /// ensure useful constants like BaseClass are defined.
    static const SIM_DopDescription	*getDopDescription();
    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(HDK_TestGeometricMultiGrid,
			GAS_SubSolver,
			"HDK Test Geometric Multi Grid",
			getDopDescription());
};
#endif