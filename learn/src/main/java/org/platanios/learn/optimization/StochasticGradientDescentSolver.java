package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;

/**
 * TODO: Add support for L1 regularization.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class StochasticGradientDescentSolver extends AbstractStochasticIterativeSolver {
    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractStochasticIterativeSolver.AbstractBuilder<T> {
        public AbstractBuilder(AbstractStochasticFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public StochasticGradientDescentSolver build() {
            return new StochasticGradientDescentSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractStochasticFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private StochasticGradientDescentSolver(AbstractBuilder<?> builder) {
        super(builder);
    }

    @Override
    public void updateDirection() {
        currentDirection = currentGradient.mult(-1);
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.mult(currentStepSize));
    }

    @Override
    public void handleBoxConstraints() {
        if (lowerBound != null)
            if (lowerBound.size() > 1)
                currentPoint.maxElementwise(lowerBound);
            else
                currentPoint.maxElementwise(lowerBound.get(0));
        if (upperBound != null)
            if (upperBound.size() > 1)
                currentPoint.minElementwise(upperBound);
            else
                currentPoint.minElementwise(upperBound.get(0));
    }
}
