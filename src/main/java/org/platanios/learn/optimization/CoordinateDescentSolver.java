package org.platanios.learn.optimization;

import org.platanios.learn.math.MathUtilities;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.function.AbstractFunction;

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
public final class CoordinateDescentSolver extends AbstractLineSearchSolver {
    private final Method method;
    private final double epsilon = Math.sqrt(MathUtilities.computeMachineEpsilonDouble());
    private final int numberOfDimensions;

    private int currentDimension = 0;
    private boolean completedCycle = false;
    private Vector cycleStartPoint;
    private Vector cycleEndPoint;

    // TODO: Add the option to set the step size initialization method.

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractLineSearchSolver.AbstractBuilder<T> {
        private Method method = Method.CYCLE_AND_JOIN_ENDPOINTS;

        public AbstractBuilder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
        }

        public T method(Method method) {
            this.method = method;
            return self();
        }

        public CoordinateDescentSolver build() {
            return new CoordinateDescentSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    /**
     * Default method is the CYCLE_AND_JOIN_ENDPOINTS method because empirically it seems to perform better than the
     * others.
     */
    private CoordinateDescentSolver(AbstractBuilder<?> builder) {
        super(builder);
        method = builder.method;
        numberOfDimensions = builder.initialPoint.size();
        cycleStartPoint = currentPoint;
    }

    @Override
    public void updateDirection() {
        method.updateDirection(this);
        // Check to see on which side along the current direction the objective function value is decreasing.
        if (!(objective.computeValue(currentPoint.add(currentDirection.mult(epsilon))) - currentObjectiveValue
                < 0)) {
            currentDirection = currentDirection.mult(-1);
        }
    }

    @Override
    public void updatePoint() {
        method.updatePoint(this);
    }

    /**
     * An enumeration of all currently supported coordinate descent methods.
     */
    public enum Method {
        /** The algorithm cycles over the coordinates (after it uses the last coordinate it goes back to the first
         * one). */
        CYCLE {
            @Override
            protected void updateDirection(CoordinateDescentSolver solver) {
                solver.currentDirection = Vectors.buildDense(solver.numberOfDimensions, 0);
                solver.currentDirection.set(solver.currentDimension, 1);
                if (solver.currentDimension >= solver.numberOfDimensions - 1) {
                    solver.currentDimension = 0;
                } else {
                    solver.currentDimension++;
                }
            }

            @Override
            protected void updatePoint(CoordinateDescentSolver solver) {
                solver.currentPoint =
                        solver.previousPoint.add(solver.currentDirection.mult(solver.currentStepSize));
            }
        },
        /** The algorithm goes back and forth over the coordinates (it uses the coordinates in the following order:
         * \(1,2,\hdots,n-1,n,n-1,\hdots,2,1,2,\hdots\)). */
        BACK_AND_FORTH {
            @Override
            protected void updateDirection(CoordinateDescentSolver solver) {
                solver.currentDirection = Vectors.buildDense(solver.numberOfDimensions, 0);
                if (solver.currentDimension < solver.numberOfDimensions) {
                    solver.currentDirection.set(solver.currentDimension, 1);
                    solver.currentDimension++;
                } else {
                    solver.currentDirection.set(2 * solver.numberOfDimensions - solver.currentDimension - 2, 1);
                    if (solver.currentDimension >= 2 * solver.numberOfDimensions - 2) {
                        solver.currentDimension = 1;
                    } else {
                        solver.currentDimension++;
                    }
                }
            }

            @Override
            protected void updatePoint(CoordinateDescentSolver solver) {
                solver.currentPoint =
                        solver.previousPoint.add(solver.currentDirection.mult(solver.currentStepSize));
            }
        },
        /** The algorithm cycles over the coordinates as with the {@link #CYCLE} restartMethod, but after each cycle completes,
         * it takes a step in the direction computed as the difference between the first point in the cycle and the last
         * point in the cycle. */
        CYCLE_AND_JOIN_ENDPOINTS {
            @Override
            protected void updateDirection(CoordinateDescentSolver solver) {
                solver.currentDirection = Vectors.buildDense(solver.numberOfDimensions, 0);
                if (!solver.completedCycle) {
                    solver.currentDirection.set(solver.currentDimension, 1);
                    if (solver.currentDimension >= solver.numberOfDimensions - 1) {
                        solver.completedCycle = true;
                        solver.currentDimension++;
                    } else {
                        solver.currentDimension++;
                    }
                } else {
                    solver.currentDirection = solver.cycleEndPoint.sub(solver.cycleStartPoint);
                    solver.currentDimension = 0;
                    solver.completedCycle = false;
                }
            }

            @Override
            protected void updatePoint(CoordinateDescentSolver solver) {
                solver.currentPoint =
                        solver.previousPoint.add(solver.currentDirection.mult(solver.currentStepSize));
                if (solver.currentDimension == 0) {
                    solver.cycleStartPoint = solver.cycleEndPoint;
                } else if (solver.currentDimension > solver.numberOfDimensions - 1) {
                    solver.cycleEndPoint = solver.currentPoint;
                }
            }
        };

        protected abstract void updateDirection(CoordinateDescentSolver solver);
        protected abstract void updatePoint(CoordinateDescentSolver solver);
    }
}
