package org.platanios.learn.optimization.linesearch;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractFunction;

/**
 * An enumeration of all possible step size initialization methods, used for computing the initial step size value for
 * iterative line search algorithms, that are currently supported by our implementation. For methods that do not produce
 * well scaled search directions, such as the steepest descent and conjugate gradient methods, it is important to use
 * current information about the problem and the algorithm to make the initial guess. For Newton and quasi-Newton
 * methods, the {@link #UNIT} step size initialization method should always be selected. This choice ensures that unit
 * step lengths are taken whenever they satisfy the termination conditions and allows the rapid rate of convergence
 * properties of these methods to take effect.
 */
public enum StepSizeInitializationMethod {
    /** Initializes the step size to a provided constant value. */
    CONSTANT {
        /** {@inheritDoc} */
        @Override
        public double computeInitialStepSize(AbstractFunction objective,
                                             Vector point,
                                             Vector direction,
                                             Vector previousPoint,
                                             Vector previousDirection,
                                             double initialStepSize,
                                             double previousStepSize) {
            return initialStepSize;
        }
    },
    /** Initializes the step size to the constant value 1. */
    UNIT {
        /** {@inheritDoc} */
        @Override
        public double computeInitialStepSize(AbstractFunction objective,
                                             Vector point,
                                             Vector direction,
                                             Vector previousPoint,
                                             Vector previousDirection,
                                             double initialStepSize,
                                             double previousStepSize) {
            return 1;
        }
    },
    /**
     * Computes a value for the initial step size (used by iterative line search algorithms) by assuming that the first
     * order change in the objective function at the current iterate/point will be the same as the one obtained in the
     * previous step.
     */
    CONSERVE_FIRST_ORDER_CHANGE {
        /** {@inheritDoc} */
        @Override
        public double computeInitialStepSize(AbstractFunction objective,
                                             Vector point,
                                             Vector direction,
                                             Vector previousPoint,
                                             Vector previousDirection,
                                             double initialStepSize,
                                             double previousStepSize) {
            // Check whether the previous direction is set (if it is not it means that we are on the first iteration of
            // the optimization algorithm).
            if (previousDirection != null) {
                return previousStepSize
                        * objective.getGradient(previousPoint).innerProduct(previousDirection)
                        /  objective.getGradient(point).innerProduct(direction);
            } else {
                return 1;
            }
        }
    },
    /**
     * Computes a value for the initial step size (used by iterative line search algorithms) by setting it to equal to
     * the minimizer of a quadratic interpolation to the current data: the objective function value at the current
     * iterate/point, the objective function value at the previous iterate/point, the objective function gradient at the
     * previous iterate/point and the previous direction used by the algorithm.
     */
    QUADRATIC_INTERPOLATION {
        /** {@inheritDoc} */
        @Override
        public double computeInitialStepSize(AbstractFunction objective,
                                             Vector point,
                                             Vector direction,
                                             Vector previousPoint,
                                             Vector previousDirection,
                                             double initialStepSize,
                                             double previousStepSize) {
            // Check whether the previous direction is set (if it is not it means that we are on the first iteration of
            // the optimization algorithm).
            if (previousDirection != null) {
                return 2 * (objective.getValue(point) - objective.getValue(previousPoint))
                        / objective.getGradient(previousPoint).innerProduct(previousDirection);
            } else {
                return 1;
            }
        }
    },
    /**
     * Computes a value for the initial step size (used by iterative line search algorithms) by setting it to equal to
     * the minimum between 1 and 1.01 times the minimizer of a quadratic interpolation to the current data: the
     * objective function value at the current iterate/point, the objective function value at the previous
     * iterate/point, the objective function gradient at the previous iterate/point and the previous direction used by
     * the algorithm.
     */
    MODIFIED_QUADRATIC_INTERPOLATION {
        /** {@inheritDoc} */
        @Override
        public double computeInitialStepSize(AbstractFunction objective,
                                             Vector point,
                                             Vector direction,
                                             Vector previousPoint,
                                             Vector previousDirection,
                                             double initialStepSize,
                                             double previousStepSize) {
            // Check whether the previous direction is set (if it is not it means that we are on the first iteration
            // of the optimization algorithm).
            if (previousDirection != null) {
                return Math.min(1, 2.02 * (objective.getValue(point) - objective.getValue(previousPoint))
                        / objective.getGradient(previousPoint).innerProduct(previousDirection));
            } else {
                return 1;
            }
        }
    };

    /**
     * Computes the initial step size using the selected method and the available data.
     *
     * @param   objective           The objective function of the optimization problem.
     * @param   point               The current iterate/point.
     * @param   direction           The current direction selected by the optimization algorithm.
     * @param   previousPoint       The iterate/point of the previous iteration.
     * @param   previousDirection   The direction used by the optimization algorithm in the previous iteration.
     * @param   initialStepSize     The initial step size value chosen by the user (if there is one - otherwise the
     *                              method selected should not be {@link #CONSTANT}. If the method selected is not
     *                              {@link #CONSTANT} then this value is ignored.
     * @param   previousStepSize    The step size used by the optimization algorithm in the previous iterations.
     * @return                      A value for the initial step size, to be used by iterative line search algorithms.
     */
    public abstract double computeInitialStepSize(AbstractFunction objective,
                                                  Vector point,
                                                  Vector direction,
                                                  Vector previousPoint,
                                                  Vector previousDirection,
                                                  double initialStepSize,
                                                  double previousStepSize);
}