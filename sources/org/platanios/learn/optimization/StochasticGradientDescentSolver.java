package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.AbstractStochasticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class StochasticGradientDescentSolver extends AbstractStochasticIterativeSolver {
    public static class Builder extends AbstractStochasticIterativeSolver.Builder<StochasticGradientDescentSolver> {
        public Builder(AbstractStochasticFunction objective, double[] initialPoint) {
            super(objective, initialPoint);
        }

        public StochasticGradientDescentSolver build() {
            return new StochasticGradientDescentSolver(this);
        }
    }

    private StochasticGradientDescentSolver(Builder builder) {
        super(builder);
    }

    @Override
    public void updateDirection() {
        currentDirection = currentGradient.multiply(-1);
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.multiply(currentStepSize));
    }
}
