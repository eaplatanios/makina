package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientDescentSolver extends AbstractLineSearchSolver {
    public static class Builder extends AbstractLineSearchSolver.Builder<GradientDescentSolver> {
        public Builder(AbstractFunction objective, double[] initialPoint) {
            super(objective, initialPoint);
        }

        public GradientDescentSolver build() {
            return new GradientDescentSolver(this);
        }
    }

    private GradientDescentSolver(Builder builder) {
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
