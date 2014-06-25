package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.linesearch.LineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SteepestDescentSolver extends AbstractSolver {
    public SteepestDescentSolver(Function objectiveFunction,
                                 double[] initialPoint) {
        super(objectiveFunction, initialPoint);
    }

    public SteepestDescentSolver(Function objectiveFunction,
                                 double[] initialPoint,
                                 LineSearch lineSearch) {
        super(objectiveFunction, initialPoint);
        setLineSearch(lineSearch);
    }

    public void updateDirection() {
        currentDirection = objectiveFunction.computeGradient(currentPoint).mapMultiply(-1);
    }

    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
    }
}
