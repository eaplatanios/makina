package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.CholeskyDecomposition;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.NonSmoothFunctionException;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.ExactLineSearch;
import org.platanios.learn.optimization.linesearch.LineSearch;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;

/**
 * TODO: Add a "set-line-search" option in the builders of all classes that inherit from this class.
 *
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractLineSearchSolver extends AbstractIterativeSolver {
    Vector currentDirection;
    Vector previousDirection;
    double currentStepSize;
    double previousStepSize;
    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    LineSearch lineSearch;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractIterativeSolver.AbstractBuilder<T> {
        protected LineSearch lineSearch;

        protected AbstractBuilder(AbstractFunction objective,
                                  Vector initialPoint) {
            super(objective, initialPoint);

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

        public T lineSearch(LineSearch lineSearch) {
            this.lineSearch = lineSearch;
            return self();
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected AbstractLineSearchSolver(AbstractBuilder<?> builder) {
        super(builder);
        lineSearch = builder.lineSearch;
        try {
            currentGradient = objective.getGradient(currentPoint);
        } catch (NonSmoothFunctionException e) {
            logger.info("The objective function being optimized is non-smooth.");
        }
        currentObjectiveValue = objective.getValue(currentPoint);
        previousPoint = currentPoint;
        previousGradient = currentGradient;
    }

    @Override
    public void performIterationUpdates() {
        previousDirection = currentDirection;
        updateDirection();
        previousStepSize = currentStepSize;
        updateStepSize();
        previousPoint = currentPoint;
        updatePoint();
        previousGradient = currentGradient;
        try {
            currentGradient = objective.getGradient(currentPoint);
        } catch (NonSmoothFunctionException e) {
            throw new UnsupportedOperationException(
                    "Line search methods cannot be used for optimizing non-smooth objective functions."
            );
        }
        if (checkForObjectiveConvergence || logObjectiveValue) {
            previousObjectiveValue = currentObjectiveValue;
            currentObjectiveValue = objective.getValue(currentPoint);
        }
    }

    public void updateStepSize() {
        try {
            currentStepSize = lineSearch.computeStepSize(currentPoint,
                                                         currentDirection,
                                                         previousPoint,
                                                         previousDirection,
                                                         previousStepSize);
        } catch (NonSmoothFunctionException ignored) {
            // TODO: I'm not sure if this exception should be ignored here.
        }
    }

    /**
     *
     *
     * Note: Care must be taken when implementing this method because the previousDirection and the previousPoint
     * variables are simply updated to point to currentDirection and currentPoint respectively, at the beginning of each
     * iteration. That means that when the new values are computed, new objects have to be instantiated for holding
     * those values.
     */
    public abstract void updateDirection();

    /**
     *
     *
     * Note: Care must be taken when implementing this method because the previousDirection and the previousPoint
     * variables are simply updated to point to currentDirection and currentPoint respectively, at the beginning of each
     * iteration. That means that when the new values are computed, new objects have to be instantiated for holding
     * those values.
     */
    public abstract void updatePoint();
}
