package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.linesearch.LineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientDescentSolver extends AbstractLineSearchSolver {
    public GradientDescentSolver(AbstractFunction objective,
                                 double[] initialPoint) {
        super(objective, initialPoint);
    }

    public GradientDescentSolver(AbstractFunction objective,
                                 double[] initialPoint,
                                 LineSearch lineSearch) {
        super(objective, initialPoint);
        setLineSearch(lineSearch);
    }

    @Override
    public void updateDirection() {
        currentGradient = objective.computeGradient(currentPoint);
        currentDirection = currentGradient.mapMultiply(-1);
    }

    @Override
    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
    }
}
