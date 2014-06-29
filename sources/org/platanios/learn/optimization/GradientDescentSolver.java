package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.linesearch.LineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientDescentSolver extends AbstractLineSearchSolver {
    public GradientDescentSolver(Function objective,
                                 double[] initialPoint) {
        super(objective, initialPoint);
    }

    public GradientDescentSolver(Function objective,
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
