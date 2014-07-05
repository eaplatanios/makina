package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientDescentSolver extends AbstractLineSearchSolver {
    public GradientDescentSolver(AbstractFunction objective,
                                 double[] initialPoint) {
        super(objective, initialPoint);
    }

    @Override
    public void updateDirection() {
        currentGradient = objective.getGradient(currentPoint);
        currentDirection = currentGradient.multiply(-1);
    }

    @Override
    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.multiply(currentStepSize));
    }
}
