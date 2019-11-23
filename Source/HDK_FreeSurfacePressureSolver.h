#ifndef HDK_FREE_SURFACE_PRESSURE_SOLVER_H
#define HDK_FREE_SURFACE_PRESSURE_SOLVER_H

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

#include <SIM/SIM_RawIndexField.h>

#include "HDK_Utilities.h"

class SIM_VectorField;
class SIM_ScalarField;

class GAS_API HDK_FreeSurfacePressureSolver : public GAS_SubSolver
{
    using SolveReal = double;
    using Vector = Eigen::VectorXd;

public:

    GET_DATA_FUNC_F(SIM_NAME_TOLERANCE, SolverTolerance);
    GET_DATA_FUNC_I("maxIterations", MaxIterations);

    GET_DATA_FUNC_B("useOldPressure", UseOldPressure);

protected:

    explicit HDK_FreeSurfacePressureSolver(const SIM_DataFactory *factory);
    virtual ~HDK_FreeSurfacePressureSolver();

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
    DECLARE_DATAFACTORY(HDK_FreeSurfacePressureSolver,
                        GAS_SubSolver,
                        "HDK Free Surface Pressure Solver",
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

    void
    buildRHS(Vector &rhsVector,
		const SIM_RawIndexField &liquidCellIndices,
		const SIM_RawIndexField &materialCellLabels,
		const SIM_VectorField &velocity,
		const SIM_VectorField *solidVelocity,
		const SIM_VectorField &validFaces,
		const std::array<SIM_RawField, 3> &cutCellWeights) const;

    void
    buildPoissonRows(std::vector<std::vector<Eigen::Triplet<SolveReal>>> &parallelPoissonElements,
			const SIM_RawIndexField &liquidCellIndices,
			const SIM_RawIndexField &materialCellLabels,
			const SIM_VectorField &validFaces,
			const SIM_RawField &liquidSurface,
			const std::array<SIM_RawField, 3> &cutCellWeights) const;

    void
    applyOldPressure(Vector &solutionVector,
			const SIM_RawField &pressure,
			const SIM_RawIndexField &liquidCellIndices) const;

    void
    buildMGBoundaryWeights(SIM_RawField &boundaryWeights,
			    const SIM_RawField &validFaces,
			    const SIM_RawField &liquidSurface,
			    const SIM_RawIndexField &liquidCellIndices,
			    const int axis) const;

    void
    applySolutionToPressure(SIM_RawField &pressure,
			    const SIM_RawIndexField &liquidCellIndices,
			    const Vector &solutionVector) const;

    void
    applyPressureGradient(SIM_RawField &velocity,
			    const SIM_RawField &validFaces,
			    const SIM_RawField &pressure,
			    const SIM_RawField &liquidSurface,
			    const SIM_RawField &cutCellWeights,
			    const SIM_RawIndexField &liquidCellIndices,
			    const int axis) const;

    void
    buildDebugDivergence(UT_VoxelArray<SolveReal> &debugDivergenceGrid,
			    const SIM_RawIndexField &materialCellLabels,
			    const SIM_VectorField &velocity,
			    const SIM_VectorField *solidVelocity,
			    const SIM_VectorField &validFaces,
			    const std::array<SIM_RawField, 3> &cutCellWeights) const;

    void
    buildMaxAndSumGrid(UT_Array<SolveReal> &tiledMaxDivergenceList,
			UT_Array<SolveReal> &tiledSumDivergenceList,
			const UT_VoxelArray<SolveReal> &debugDivergenceGrid,
			const SIM_RawIndexField &materialCellLabels) const;    
};

#endif