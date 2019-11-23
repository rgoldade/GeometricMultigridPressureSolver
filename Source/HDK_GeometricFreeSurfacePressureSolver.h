#ifndef HDK_FREE_SURFACE_PRESSURE_SOLVER_H
#define HDK_FREE_SURFACE_PRESSURE_SOLVER_H

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

#include <SIM/SIM_RawIndexField.h>

#include "HDK_Utilities.h"

class SIM_VectorField;
class SIM_ScalarField;

class GAS_API HDK_GeometricFreeSurfacePressureSolver : public GAS_SubSolver
{
    using StoreReal = double;
    using SolveReal = double;

public:

    GET_DATA_FUNC_F(SIM_NAME_TOLERANCE, SolverTolerance);
    GET_DATA_FUNC_I("maxIterations", MaxSolverIterations);

    GET_DATA_FUNC_B("useMGPreconditioner", UseMGPreconditioner);

    GET_DATA_FUNC_B("useOldPressure", UseOldPressure);

protected:

    explicit HDK_GeometricFreeSurfacePressureSolver(const SIM_DataFactory *factory);
    virtual ~HDK_GeometricFreeSurfacePressureSolver();

    // The overloaded callback that GAS_SubSolver will invoke to
    // perform our actual computation.  We are giving a single object
    // at a time to work on.
    virtual bool solveGasSubclass(SIM_Engine &engine,
				    SIM_Object *obj,
				    SIM_Time time,
				    SIM_Time timestep);
private:

    // We define this to be a DOP_Auto node which means we do not
    // need to implement a DOP_Node derivative for this data.  Instead,
    // this description is used to define the interface.
    static const SIM_DopDescription *getDopDescription();
    
    /// These macros are necessary to bind our node to the factory and
    /// ensure useful constants like BaseClass are defined.
    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(HDK_GeometricFreeSurfacePressureSolver,
                        GAS_SubSolver,
                        "HDK Geometric Free Surface Pressure Solver",
                        getDopDescription());

    ////////////////////////////////////////////
    //
    // Solver methods
    //
    ////////////////////////////////////////////

    void
    buildValidFaces(SIM_VectorField &validFaces,
		    const SIM_RawIndexField &materialCellLabels,
		    const std::array<SIM_RawField, 3> &cutCellWeights) const;

    std::pair<UT_Vector3I, int>
    buildMGDomainLabels(UT_VoxelArray<int> &domainCellLabels,
			const SIM_RawIndexField &materialCellLabels) const;

    void
    buildRHS(UT_VoxelArray<StoreReal> &rhsGrid,
		const SIM_RawIndexField &materialCellLabels,
		const UT_VoxelArray<int> &domainCellLabels,
		const SIM_VectorField &velocity,
		const SIM_VectorField *solidVelocity,
		const SIM_VectorField &validFaces,
		const std::array<SIM_RawField, 3> &cutCellWeights,
		const UT_Vector3I exteriorOffset) const;

    void
    applyOldPressure(UT_VoxelArray<StoreReal> &solutionGrid,
			const SIM_RawField &pressure,
			const SIM_RawIndexField &materialCellLabels,
			const UT_VoxelArray<int> &domainCellLabels,
			const UT_Vector3I &exteriorOffset) const;

    void
    buildMGBoundaryWeights(UT_VoxelArray<StoreReal> &boundaryWeights,
			    const SIM_RawField &cutCellWeights,
			    const SIM_RawField &validFaces,
			    const SIM_RawField &liquidSurface,
			    const SIM_RawIndexField &materialCellLabels,
			    const UT_VoxelArray<int> &domainCellLabels,
			    const UT_Vector3I &exteriorOffset,
			    const int axis) const;

    void
    setBoundaryDomainLabels(UT_VoxelArray<int> &domainCellLabels,
			    const std::array<UT_VoxelArray<StoreReal>, 3> &boundaryWeights) const;

    void
    applySolutionToPressure(SIM_RawField &pressure,
			    const UT_VoxelArray<StoreReal> &solutionGrid,
			    const SIM_RawIndexField &materialCellLabels,
			    const UT_VoxelArray<int> &domainCellLabels,
			    const UT_Vector3I &exteriorOffset) const;

    void
    applyPressureGradient(SIM_RawField &velocity,
			    const SIM_RawField &validFaces,
			    const SIM_RawField &pressure,
			    const SIM_RawField &liquidSurface,
			    const SIM_RawField &cutCellWeights,
			    const SIM_RawIndexField &materialCellLabels,
			    const int axis) const;

    void
    buildDebugDivergence(UT_VoxelArray<StoreReal> &debugDivergenceGrid,
			    const SIM_RawIndexField &materialCellLabels,
			    const SIM_VectorField &velocity,
			    const SIM_VectorField *solidVelocity,
			    const SIM_VectorField &validFaces,
			    const std::array<SIM_RawField, 3> &cutCellWeights) const;

    void
    buildMaxAndSumGrid(UT_Array<SolveReal> &tiledMaxDivergenceList,
			UT_Array<SolveReal> &tiledSumDivergenceList,
			const UT_VoxelArray<StoreReal> &debugDivergenceGrid,
			const SIM_RawIndexField &materialCellLabels) const;

};

#endif