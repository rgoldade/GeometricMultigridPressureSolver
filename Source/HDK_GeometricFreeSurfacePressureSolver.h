#ifndef HDK_GEOMETRIC_FREE_SURFACE_PRESSURE_SOLVER_H
#define HDK_GEOMETRIC_FREE_SURFACE_PRESSURE_SOLVER_H

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

#include <SIM/SIM_RawIndexField.h>

#include "HDK_Utilities.h"

class SIM_VectorField;
class SIM_ScalarField;

class GAS_API HDK_GeometricFreeSurfacePressureSolver : public GAS_SubSolver
{
    using MaterialLabels = HDK::Utilities::FreeSurfaceMaterialLabels;

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
		    const std::array<const SIM_RawField *, 3> &cutCellWeights) const;

    void
    buildMGDomainLabels(UT_VoxelArray<int> &mgDomainCellLabels,
			const SIM_RawIndexField &materialCellLabels) const;

    void
    buildMGBoundaryWeights(UT_VoxelArray<SolveReal> &mgBoundaryWeights,
			    const SIM_RawField &cutCellWeights,
			    const SIM_RawField &liquidSurface,
			    const SIM_RawField &validFaces,
			    const SIM_RawIndexField &materialCellLabels,
			    const UT_VoxelArray<int> &mgDomainCellLabels,
			    const int axis) const;

    void
    buildRHS(UT_VoxelArray<StoreReal> &rhsGrid,
		const SIM_RawIndexField &materialCellLabels,
		const SIM_VectorField &velocity,
		const SIM_VectorField *solidVelocity,
		const std::array<const SIM_RawField *, 3> &cutCellWeights,
		const UT_VoxelArray<int> &mgDomainCellLabels,
		const UT_Vector3I mgExpandedOffset) const;

    void
    applyOldPressure(UT_VoxelArray<StoreReal> &solutionGrid,
			const SIM_RawField &pressure,
			const SIM_RawIndexField &materialCellLabels,
			const UT_VoxelArray<int> &mgDomainCellLabels,
			const UT_Vector3I &mgExpandedOffset) const;

    void
    applySolutionToPressure(SIM_RawField &pressure,
			    const SIM_RawIndexField &materialCellLabels,
			    const UT_VoxelArray<int> &mgDomainCellLabels,
			    const UT_VoxelArray<StoreReal> &solutionGrid,
			    const UT_Vector3I &mgExpandedOffset) const;

    void
    applyPressureGradient(SIM_RawField &velocity,
			    const SIM_RawField &cutCellWeights,
			    const SIM_RawField &liquidSurface,
			    const SIM_RawField &pressure,
			    const SIM_RawField &validFaces,
			    const SIM_RawIndexField &materialCellLabels,
			    const int axis) const;

    void
    computeResultingDivergence(UT_Array<SolveReal> &parallelAccumulatedDivergence,
				UT_Array<SolveReal> &parallelCellCount,	
				UT_Array<SolveReal> &parallelMaxDivergence,				
				const SIM_RawIndexField &materialCellLabels,
				const SIM_VectorField &velocity,
				const SIM_VectorField *solidVelocity,
				const std::array<const SIM_RawField *, 3> &cutCellWeights) const;
};

#endif