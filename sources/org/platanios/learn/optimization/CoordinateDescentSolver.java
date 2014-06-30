package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;

/**
 * This is a derivative-free optimization algorithm.
 *
 * @author Emmanouil Antonios Platanios
 */
public class CoordinateDescentSolver extends AbstractLineSearchSolver {
    private final double epsilon = Math.sqrt(calculateMachineEpsilonDouble());
    private final int numberOfDimensions;

    private int currentDimension = 0;

    public CoordinateDescentSolver(AbstractFunction objective,
                                   double[] initialPoint) {
        super(objective, initialPoint);
        numberOfDimensions = initialPoint.length;
        StrongWolfeInterpolationLineSearch lineSearch = new StrongWolfeInterpolationLineSearch(objective,
                                                                                               1e-4,
                                                                                               0.9,
                                                                                               10);
        lineSearch.setStepSizeInitializationMethod(StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE);
        setLineSearch(lineSearch);
    }

    @Override
    public void updateDirection() {
        currentDirection = new ArrayRealVector(numberOfDimensions, 0);
        currentDirection.setEntry(currentDimension, 1);
        // Check to see on which side along the current direction the objective function value is decreasing.
        if (!(objective.computeValue(currentPoint.add(currentDirection.mapMultiply(epsilon))) - currentObjectiveValue
                < 0)) {
            currentDirection = currentDirection.mapMultiply(-1);
        }

        if (currentDimension == numberOfDimensions - 1) {
            currentDimension = 0;
        } else {
            currentDimension++;
        }
    }

    @Override
    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
    }

    // TODO: Move this method to a utils class - it is also used in the DerivativesApproximation class.
    private static double calculateMachineEpsilonDouble() {
        double epsilon = 1;
        while (1 + epsilon / 2 > 1.0) {
            epsilon /= 2;
        }
        return epsilon;
    }
}
