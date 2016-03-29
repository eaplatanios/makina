package org.platanios.optimization;

import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;
import org.platanios.optimization.function.AbstractStochasticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class AdaptiveGradientSolver extends AbstractStochasticIterativeSolver {
    private final double epsilon = Math.sqrt(Double.MIN_VALUE);

    private Vector sumOfGradients;
    private Vector squareRootOfSumOfGradientSquares;

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
        public Builder(AbstractStochasticFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    protected AdaptiveGradientSolver(AbstractBuilder<?> builder) {
        super(builder);

        sumOfGradients = Vectors.build(currentGradient.size(), currentGradient.type());
        squareRootOfSumOfGradientSquares = Vectors.build(currentGradient.size(), currentGradient.type());
    }

    @Override
    public void updateDirection() {
        squareRootOfSumOfGradientSquares.hypotenuseFastInPlace(currentGradient);
        if (useL1Regularization) {
            currentDirection = sumOfGradients
                    .addInPlace(currentGradient)
                    .mapDivElementwise(x -> Math.abs(x) / (currentIteration + 1) <= l1RegularizationWeight ?
                            0.0 : - currentIteration * Math.signum(x)
                                               * (Math.abs(x) / (currentIteration + 1) - l1RegularizationWeight),
                                          squareRootOfSumOfGradientSquares.add(epsilon));
        } else {
            currentDirection = currentGradient.divElementwise(squareRootOfSumOfGradientSquares.add(epsilon)).mult(-1);
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
