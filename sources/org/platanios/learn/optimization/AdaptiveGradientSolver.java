package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class AdaptiveGradientSolver extends AbstractStochasticIterativeSolver {
    private Vector sumOfGradientSquares;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractStochasticIterativeSolver.AbstractBuilder<T> {
        public AbstractBuilder(AbstractStochasticFunction objective, double[] initialPoint) {
            super(objective, initialPoint);
        }

        public AdaptiveGradientSolver build() {
            return new AdaptiveGradientSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractStochasticFunction objective,
                       double[] initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private AdaptiveGradientSolver(AbstractBuilder<?> builder) {
        super(builder);
    }

    @Override
    public void updateDirection() {
        if (currentIteration == 0) {
            sumOfGradientSquares = currentGradient.computeFunctionResult(gradient -> Math.pow(gradient, 2)).copy();
        } else {
            sumOfGradientSquares.addEquals(currentGradient.computeFunctionResult(gradient -> Math.pow(gradient, 2)));
        }
        currentDirection =
                currentGradient.multiply(-1).divideElementwise(sumOfGradientSquares.computeFunctionResult(Math::sqrt));
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.multiply(currentStepSize));
    }
}
