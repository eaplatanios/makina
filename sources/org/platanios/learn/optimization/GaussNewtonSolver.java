package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.AbstractLeastSquaresFunction;
import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;

/**
 * This is a method for solving nonlinear least squares problems.
 *
 * @author Emmanouil Antonios Platanios
 */
public class GaussNewtonSolver extends AbstractLineSearchSolver {
    private final LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod;
    // TODO: Add a way to control the preconditioning method of the subproblem solver.

    public static class Builder {
        // Required parameters
        private final AbstractLeastSquaresFunction objective;
        private final double[] initialPoint;

        // Optional parameters - Initialized to default values
        private LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod =
                LinearLeastSquaresSolver.Method.SINGULAR_VALUE_DECOMPOSITION;

        public Builder(AbstractLeastSquaresFunction objective, double[] initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }

        public Builder linearLeastSquaresSubproblemMethod(
                LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod
        ) {
            this.linearLeastSquaresSubproblemMethod = linearLeastSquaresSubproblemMethod;
            return this;
        }

        public GaussNewtonSolver build() {
            return new GaussNewtonSolver(this);
        }
    }

    private GaussNewtonSolver(Builder builder) {
        super(builder.objective, builder.initialPoint);
        this.linearLeastSquaresSubproblemMethod = builder.linearLeastSquaresSubproblemMethod;
        StrongWolfeInterpolationLineSearch lineSearch = new StrongWolfeInterpolationLineSearch(objective, 1e-4, 0.9, 1);
        lineSearch.setStepSizeInitializationMethod(StepSizeInitializationMethod.UNIT);
        setLineSearch(lineSearch);
    }

    @Override
    public void updateDirection() {
        LinearLeastSquaresSolver linearLeastSquaresSubproblemSolver =
                new LinearLeastSquaresSolver.Builder(new LinearLeastSquaresFunction(
                        ((AbstractLeastSquaresFunction) objective).computeJacobian(currentPoint),
                        ((AbstractLeastSquaresFunction) objective).computeResiduals(currentPoint).multiply(-1)
                )).method(linearLeastSquaresSubproblemMethod).build();
        currentDirection = linearLeastSquaresSubproblemSolver.solve();
    }

    @Override
    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.multiply(currentStepSize));
    }
}
