package makina.optimization;

import makina.math.matrix.Vector;
import makina.optimization.function.AbstractStochasticFunction;

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
                currentPoint.maxElementwiseInPlace(lowerBound);
            else
                currentPoint.maxElementwiseInPlace(lowerBound.get(0));
        if (upperBound != null)
            if (upperBound.size() > 1)
                currentPoint.minElementwiseInPlace(upperBound);
            else
                currentPoint.minElementwiseInPlace(upperBound.get(0));
    }
}
