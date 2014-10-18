package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.function.AbstractStochasticFunction;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class AdaptiveGradientSolver extends AbstractStochasticIterativeSolver {
    private final double epsilon = Math.sqrt(Double.MIN_VALUE);

    private Vector sumOfGradients;
    private Vector sumOfGradientSquares;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractStochasticIterativeSolver.AbstractBuilder<T> {
        public AbstractBuilder(AbstractStochasticFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public AdaptiveGradientSolver build() {
            return new AdaptiveGradientSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractStochasticFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private AdaptiveGradientSolver(AbstractBuilder<?> builder) {
        super(builder);

        sumOfGradients = Vectors.build(currentGradient.size(), currentGradient.type());
        sumOfGradientSquares = Vectors.build(currentGradient.size(), currentGradient.type()).add(epsilon);
    }

    @Override
    public void updateDirection() {
        sumOfGradientSquares.addInPlace(currentGradient.map(gradient -> Math.pow(gradient, 2)));
        if (useL1Regularization) {
            currentDirection = sumOfGradients
                    .addInPlace(currentGradient)
                    .map(x -> Math.abs(x) / (currentIteration + 1) <= l1RegularizationWeight ?
                            0.0 : - Math.signum(x) * (Math.abs(x) / (currentIteration + 1) - l1RegularizationWeight))
                    .divElementwise(sumOfGradientSquares.map(Math::sqrt))
                    .mult(currentIteration);
        } else {
            currentDirection = currentGradient.divElementwise(sumOfGradientSquares.map(Math::sqrt)).mult(-1);
        }
    }

    @Override
    public void updatePoint() {
        if (!useL1Regularization) {
            currentPoint = previousPoint.add(currentDirection.mult(currentStepSize));
        } else {
            currentPoint = currentDirection.mult(currentStepSize);
        }
    }
}
