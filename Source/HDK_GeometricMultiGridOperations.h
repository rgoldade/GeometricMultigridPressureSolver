#ifndef HDK_GEOMETRIC_MULTI_GRID_OPERATIONS_H
#define HDK_GEOMETRIC_MULTI_GRID_OPERATIONS_H

#include <UT/UT_VoxelArray.h>

namespace HDK::GeometricMultiGridOperations{

    enum CellLabels {INTERIOR_CELL, EXTERIOR_CELL, DIRICHLET_CELL, BOUNDARY_CELL};

    using StoreReal = float;
    using SolveReal = double;

    void dampedJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
					const UT_VoxelArray<StoreReal> &rhs,
					const UT_VoxelArray<int> &cellLabels,
					const SolveReal dx);

    void dampedJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
					const UT_VoxelArray<StoreReal> &rhs,
					const UT_VoxelArray<int> &cellLabels,
					const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
					const SolveReal dx);

    // Jacobi smoothing along domain boundaries

    void dampedJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
					const UT_VoxelArray<StoreReal> &rhs,
					const UT_VoxelArray<int> &cellLabels,
					const UT_Array<UT_Vector3I> &boundaryCells,
					const SolveReal dx);

    void dampedJacobiPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
					const UT_VoxelArray<StoreReal> &rhs,
					const UT_VoxelArray<int> &cellLabels,
					const UT_Array<UT_Vector3I> &boundaryCells,
					const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
					const SolveReal dx);

    void tiledGaussSeidelPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
					    const UT_VoxelArray<StoreReal> &rhs,
					    const UT_VoxelArray<int> &cellLabels,
					    const SolveReal dx,
					    const bool doSmoothOddTiles,
					    const bool doSmoothForward);

    void tiledGaussSeidelPoissonSmoother(UT_VoxelArray<StoreReal> &solution,
					    const UT_VoxelArray<StoreReal> &rhs,
					    const UT_VoxelArray<int> &cellLabels,
					    const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
					    const SolveReal dx,
					    const bool doSmoothOddTiles,
					    const bool doSmoothForward);

    // TODO: add tiled red-black gauss seidel

    void applyPoissonMatrix(UT_VoxelArray<StoreReal> &destination,
			    const UT_VoxelArray<StoreReal> &source,
			    const UT_VoxelArray<int> &cellLabels,
			    const SolveReal dx);

    void applyPoissonMatrix(UT_VoxelArray<StoreReal> &destination,
			    const UT_VoxelArray<StoreReal> &source,
			    const UT_VoxelArray<int> &cellLabels,
			    const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
			    const SolveReal dx);

    void computePoissonResidual(UT_VoxelArray<StoreReal> &residual,
				const UT_VoxelArray<StoreReal> &solution,
				const UT_VoxelArray<StoreReal> &rhs,
				const UT_VoxelArray<int> &cellLabels,
				const SolveReal dx);

    void computePoissonResidual(UT_VoxelArray<StoreReal> &residual,
				const UT_VoxelArray<StoreReal> &solution,
				const UT_VoxelArray<StoreReal> &rhs,
				const UT_VoxelArray<int> &cellLabels,
				const UT_VoxelArray<StoreReal> (&gradientWeights)[3],
				const SolveReal dx);

    void downsample(UT_VoxelArray<StoreReal> &destination,
		    const UT_VoxelArray<StoreReal> &source,
		    const UT_VoxelArray<int> &destinationCellLabels,
		    const UT_VoxelArray<int> &sourceCellLabels);

    void upsample(UT_VoxelArray<StoreReal> &destination,
		    const UT_VoxelArray<StoreReal> &source,
		    const UT_VoxelArray<int> &destinationCellLabels,
		    const UT_VoxelArray<int> &sourceCellLabels);

    void upsampleAndAdd(UT_VoxelArray<StoreReal> &destination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &destinationCellLabels,
			const UT_VoxelArray<int> &sourceCellLabels);

    UT_VoxelArray<int> buildCoarseCellLabels(const UT_VoxelArray<int> &sourceCellLabels);

    UT_Array<UT_Vector3I> buildBoundaryCells(const UT_VoxelArray<int> &sourceCellLabels,
						const int boundaryWidth);

    bool unitTestCoarsening(const UT_VoxelArray<int> &coarseCellLabels,
			    const UT_VoxelArray<int> &fineCellLabels);

    StoreReal dotProduct(const UT_VoxelArray<StoreReal> &vectorA,
			    const UT_VoxelArray<StoreReal> &vectorB,
			    const UT_VoxelArray<int> &cellLabels);

    void addToVector(UT_VoxelArray<StoreReal> &destination,
			const UT_VoxelArray<StoreReal> &source,
			const UT_VoxelArray<int> &cellLabels);

    StoreReal l2Norm(const UT_VoxelArray<StoreReal> &vector,
			const UT_VoxelArray<int> &cellLabels);

    StoreReal infNorm(const UT_VoxelArray<StoreReal> &vector,
			const UT_VoxelArray<int> &cellLabels);
}

#endif