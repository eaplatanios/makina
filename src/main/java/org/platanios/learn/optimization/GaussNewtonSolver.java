package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;
import org.platanios.learn.optimization.function.AbstractLeastSquaresFunction;

/**
 * This is a method for solving nonlinear least squares problems.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class GaussNewtonSolver extends AbstractLineSearchSolver {
    private final LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod;
    // TODO: Add a way to control the preconditioning method of the subproblem solver.

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractLineSearchSolver.AbstractBuilder<T> {
        private LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod =
                LinearLeastSquaresSolver.Method.SINGULAR_VALUE_DECOMPOSITION;

        public AbstractBuilder(AbstractLeastSquaresFunction objective, double[] initialPoint) {
            super(objective, initialPoint);
        }

        public T linearLeastSquaresSubproblemMethod(
                LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod
        ) {
            this.linearLeastSquaresSubproblemMethod = linearLeastSquaresSubproblemMethod;
            return self();
        }

        public GaussNewtonSolver build() {
            return new GaussNewtonSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractLeastSquaresFunction objective,
                       double[] initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private GaussNewtonSolver(AbstractBuilder<?> builder) {
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
