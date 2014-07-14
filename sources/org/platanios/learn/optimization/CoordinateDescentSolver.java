package org.platanios.learn.optimization;

import org.platanios.learn.math.Utilities;
import org.platanios.learn.math.matrix.Vector;
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
    private final Method method;
    private final double epsilon = Math.sqrt(Utilities.computeMachineEpsilonDouble());
    private final int numberOfDimensions;

    private int currentDimension = 0;
    private boolean completedCycle = false;
    private Vector cycleStartPoint;
    private Vector cycleEndPoint;

    // TODO: Add the option to set the step size initialization method.

    public static class Builder extends AbstractLineSearchSolver.Builder<CoordinateDescentSolver> {
        private Method method = Method.CYCLE_AND_JOIN_ENDPOINTS;

        public Builder(AbstractFunction objective, double[] initialPoint) {
            super(objective, initialPoint);
        }

        public Builder method(Method method) {
            this.method = method;
            return this;
        }

        public CoordinateDescentSolver build() {
            return new CoordinateDescentSolver(this);
        }
    }

    /**
     * Default method is the CYCLE_AND_JOIN_ENDPOINTS method because empirically it seems to perform better than the
     * others.
     */
    private CoordinateDescentSolver(Builder builder) {
        super(builder);
        this.method = builder.method;
        numberOfDimensions = builder.initialPoint.length;
        cycleStartPoint = currentPoint;
    }

    @Override
    public void updateDirection() {
        currentDirection = new Vector(numberOfDimensions, 0);

        switch (method) {
            case CYCLE:
                currentDirection.setElement(currentDimension, 1);
                if (currentDimension >= numberOfDimensions - 1) {
                    currentDimension = 0;
                } else {
                    currentDimension++;
                }
                break;
            case BACK_AND_FORTH:
                if (currentDimension < numberOfDimensions) {
                    currentDirection.setElement(currentDimension, 1);
                    currentDimension++;
                } else {
                    currentDirection.setElement(2 * numberOfDimensions - currentDimension - 2, 1);
                    if (currentDimension >= 2 * numberOfDimensions - 2) {
                        currentDimension = 1;
                    } else {
                        currentDimension++;
                    }
                }
                break;
            case CYCLE_AND_JOIN_ENDPOINTS:
                if (!completedCycle) {
                    currentDirection.setElement(currentDimension, 1);
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
        if (!(objective.computeValue(previousPoint.add(currentDirection.multiply(epsilon))) - currentObjectiveValue
                < 0)) {
            currentDirection = currentDirection.multiply(-1);
        }
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.multiply(currentStepSize));

        if (method == Method.CYCLE_AND_JOIN_ENDPOINTS) {
            if (currentDimension == 0) {
                cycleStartPoint = cycleEndPoint;
            } else if (currentDimension > numberOfDimensions - 1) {
                cycleEndPoint = currentPoint;
            }
        }
    }

    /**
     * An enumeration of all currently supported coordinate descent methods.
     */
    public enum Method {
        /** The algorithm cycles over the coordinates (after it uses the last coordinate it goes back to the first
         * one). */
        CYCLE,
        /** The algorithm goes back and forth over the coordinates (it uses the coordinates in the following order:
         * \(1,2,\hdots,n-1,n,n-1,\hdots,2,1,2,\hdots\)). */
        BACK_AND_FORTH,
        /** The algorithm cycles over the coordinates as with the {@link #CYCLE} restartMethod, but after each cycle completes,
         * it takes a step in the direction computed as the difference between the first point in the cycle and the last
         * point in the cycle. */
        CYCLE_AND_JOIN_ENDPOINTS
    }
}
