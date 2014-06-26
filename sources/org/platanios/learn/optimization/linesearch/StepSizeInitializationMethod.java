package org.platanios.learn.optimization.linesearch;

/**
 * An enumeration of all possible step size initialization methods, used for computing the initial step size value for
 * iterative line search algorithms, that are currently supported by our implementation. For methods that do not produce
 * well scaled search directions, such as the steepest descent and conjugate gradient methods, it is important to use
 * current information about the problem and the algorithm to make the initial guess. For Newton and quasi-Newton
 * methods, the UNIT step size initialization method should always be selected. This choice ensures that unit step
 * lengths are taken whenever they satisfy the termination conditions and allows the rapid rate of convergence
 * properties of these methods to take effect.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum StepSizeInitializationMethod {
    /** Initialize the step size to a provided constant value. */
    CONSTANT,
    /** Initialize the step size to the constant value 1. */
    UNIT,
    /** Assume that the first order change in the objective function at the current iterate/point will be the same as
     * the one obtained in the previous step. */
    CONSERVE_FIRST_ORDER_CHANGE,
    /** Set the initial step size to the minimizer of a quadratic interpolation to the current data: the objective
     * function value at the current iterate/point, the objective function value at the previous iterate/point, the
     * objective function gradient at the previous iterate/point and the previous direction used by the algorithm. */
    QUADRATIC_INTERPOLATION,
    /** Set the initial step size to the minimum between 1 and 1.01 times the minimizer of a quadratic interpolation to
     * the current data: the objective function value at the current iterate/point, the objective function value at the
     * previous iterate/point, the objective function gradient at the previous iterate/point and the previous direction
     * used by the algorithm. This makes sure that the unit step length will eventually always be tried and accepted and
     * the superlinear convergence properties of Newton and quasi-Newton methods will be observed. */
    MODIFIED_QUADRATIC_INTERPOLATION
}
