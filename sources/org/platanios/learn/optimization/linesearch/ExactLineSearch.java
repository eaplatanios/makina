package org.platanios.learn.optimization.linesearch;

import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ExactLineSearch implements LineSearch {
    private final QuadraticFunction objectiveFunction;

    // TODO: Implement for linear function as well, apart from only quadratic.
    public ExactLineSearch(QuadraticFunction objectiveFunction) {
        this.objectiveFunction = objectiveFunction;
    }

    public double computeStepSize(RealVector currentPoint,
                                  RealVector direction) {
        RealVector objectiveFunctionGradientAtCurrentPoint = objectiveFunction.computeGradient(currentPoint);
        double stepSize = - 0.5 * objectiveFunctionGradientAtCurrentPoint.dotProduct(direction);
        stepSize /= objectiveFunction.getQ().preMultiply(direction).dotProduct(direction);
        return stepSize;
    }
}
