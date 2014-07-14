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

    public static class Builder extends AbstractLineSearchSolver.Builder<GaussNewtonSolver> {
        private LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod =
                LinearLeastSquaresSolver.Method.SINGULAR_VALUE_DECOMPOSITION;

        public Builder(AbstractLeastSquaresFunction objective, double[] initialPoint) {
            super(objective, initialPoint);
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
        super(builder);
        linearLeastSquaresSubproblemMethod = builder.linearLeastSquaresSubproblemMethod;
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
        currentPoint = previousPoint.add(currentDirection.multiply(currentStepSize));
    }
}
