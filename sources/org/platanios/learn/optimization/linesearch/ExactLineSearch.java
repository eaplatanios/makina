package org.platanios.learn.optimization.linesearch;

import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ExactLineSearch implements LineSearch {
    private final QuadraticFunction objective;

    // TODO: Implement for linear function as well, apart from only quadratic.
    public ExactLineSearch(QuadraticFunction objective) {
        this.objective = objective;
    }

    public double computeStepSize(RealVector currentPoint,
                                  RealVector direction) {
        RealVector objectiveFunctionGradientAtCurrentPoint = objective.computeGradient(currentPoint);
        double stepSize = - 0.5 * objectiveFunctionGradientAtCurrentPoint.dotProduct(direction);
        stepSize /= objective.getQ().preMultiply(direction).dotProduct(direction);
        return stepSize;
    }
}
