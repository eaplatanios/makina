package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.CholeskyDecomposition;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;

/**
 * TODO: Add a "set-line-search" option in the builders of all classes that inherit from this class.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractLineSearchSolver extends AbstractIterativeSolver {
    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    LineSearch lineSearch;

    public static abstract class Builder<T extends AbstractLineSearchSolver> {
        // Required parameters
        protected final AbstractFunction objective;
        protected final double[] initialPoint;

        // Optional parameters - Initialized to default values
        protected LineSearch lineSearch;

        protected Builder(AbstractFunction objective,
                          double[] initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;

            if (objective instanceof QuadraticFunction) {
                Matrix quadraticFactorMatrix = ((QuadraticFunction) objective).getA();
                CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(quadraticFactorMatrix);
                if (choleskyDecomposition.isSymmetricAndPositiveDefinite()) {
                    lineSearch = new ExactLineSearch((QuadraticFunction) objective);
                    return;
                }
            }

            lineSearch = new StrongWolfeInterpolationLineSearch(objective, 1e-4, 0.9, 10);
            ((StrongWolfeInterpolationLineSearch) lineSearch)
                    .setStepSizeInitializationMethod(StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE);
        }

        public Builder<T> lineSearch(LineSearch lineSearch) {
            this.lineSearch = lineSearch;
            return this;
        }

        public abstract T build();
    }

    AbstractLineSearchSolver(Builder<?> builder) {
        super(builder.objective, builder.initialPoint);
        this.lineSearch = builder.lineSearch;
    }

    @Override
    public void iterationUpdate() {
        previousGradient = currentGradient;
        previousDirection = currentDirection;
        updateDirection();
        previousStepSize = currentStepSize;
        updateStepSize();
        previousPoint = currentPoint;
        previousObjectiveValue = currentObjectiveValue;
        updatePoint();
        currentObjectiveValue = objective.getValue(currentPoint);
    }

    public void updateStepSize() {
        currentStepSize = lineSearch.computeStepSize(currentPoint,
                                                     currentDirection,
                                                     previousPoint,
                                                     previousDirection,
                                                     previousStepSize);
    }

    public abstract void updateDirection();
    public abstract void updatePoint();
}
