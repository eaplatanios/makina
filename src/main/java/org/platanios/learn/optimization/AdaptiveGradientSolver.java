package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.AbstractStochasticFunction;
import org.platanios.learn.math.matrix.Vector;

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
            sumOfGradientSquares = currentGradient.map(gradient -> Math.pow(gradient, 2)).copy();
        } else {
            sumOfGradientSquares.addInPlace(currentGradient.map(gradient -> Math.pow(gradient, 2)));
        }
        currentDirection =
                currentGradient.mult(-1).divElementwise(sumOfGradientSquares.map(Math::sqrt));
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.mult(currentStepSize));
    }
}
