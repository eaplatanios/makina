package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.linesearch.LineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GradientDescentSolver extends AbstractSolver {
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

    public void updateDirection() {
        currentDirection = objective.computeGradient(currentPoint).mapMultiply(-1);
    }

    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
    }
}