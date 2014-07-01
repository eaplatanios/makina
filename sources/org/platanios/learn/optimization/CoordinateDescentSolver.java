package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.math.Utilities;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * This is a derivative-free optimization algorithm.
 *
 * TODO: Implement a pattern-search solver.
 * TODO: Implement the DFO conjugate direction solver.
 * TODO: Implement the Nelder-Mead solver.
 * TODO: Implement the implicit filtering solver.
 *
 * @author Emmanouil Antonios Platanios
 */
public class CoordinateDescentSolver extends AbstractLineSearchSolver {
    private final double epsilon = Math.sqrt(Utilities.calculateMachineEpsilonDouble());
    private final int numberOfDimensions;

    private CoordinateDescentMethod method = CoordinateDescentMethod.CYCLE_AND_JOIN_ENDPOINTS;
    private int currentDimension = 0;
    private boolean completedCycle = false;
    private RealVector cycleStartPoint;
    private RealVector cycleEndPoint;

    /**
     * Default method is the CYCLE_AND_JOIN_ENDPOINTS method because empirically it seems to perform better than the
     * others.
     *
     * @param objective
     * @param initialPoint
     */
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
        cycleStartPoint = currentPoint;
    }

    @Override
    public void updateDirection() {
        currentDirection = new ArrayRealVector(numberOfDimensions, 0);

        switch (method) {
            case CYCLE:
                currentDirection.setEntry(currentDimension, 1);
                if (currentDimension >= numberOfDimensions - 1) {
                    currentDimension = 0;
                } else {
                    currentDimension++;
                }
                break;
            case BACK_AND_FORTH:
                if (currentDimension < numberOfDimensions) {
                    currentDirection.setEntry(currentDimension, 1);
                    currentDimension++;
                } else {
                    currentDirection.setEntry(2 * numberOfDimensions - currentDimension - 2, 1);
                    if (currentDimension >= 2 * numberOfDimensions - 2) {
                        currentDimension = 1;
                    } else {
                        currentDimension++;
                    }
                }
                break;
            case CYCLE_AND_JOIN_ENDPOINTS:
                if (!completedCycle) {
                    currentDirection.setEntry(currentDimension, 1);
                    if (currentDimension >= numberOfDimensions - 1) {
                        completedCycle = true;
                        currentDimension++;
                    } else {
                        currentDimension++;
                    }
                } else {
                    currentDirection = cycleEndPoint.subtract(cycleStartPoint);
                    currentDimension = 0;
                    completedCycle = false;
                }
                break;
            default:
                throw new NotImplementedException();
        }

        // Check to see on which side along the current direction the objective function value is decreasing.
        if (!(objective.computeValue(currentPoint.add(currentDirection.mapMultiply(epsilon))) - currentObjectiveValue
                < 0)) {
            currentDirection = currentDirection.mapMultiply(-1);
        }
    }

    @Override
    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));

        if (method == CoordinateDescentMethod.CYCLE_AND_JOIN_ENDPOINTS) {
            if (currentDimension == 0) {
                cycleStartPoint = cycleEndPoint;
            } else if (currentDimension > numberOfDimensions - 1) {
                cycleEndPoint = currentPoint;
            }
        }
    }

    public CoordinateDescentMethod getMethod() {
        return method;
    }

    public void setMethod(CoordinateDescentMethod method) {
        this.method = method;
    }
}
