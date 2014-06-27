package org.platanios.learn.optimization.linesearch;

import org.apache.commons.math3.linear.RealVector;

/**
 * An implementation of several step size initialization methods used for computing the initial step size value for
 * iterative line search algorithms. For methods that do not produce well scaled search directions, such as the steepest
 * descent and conjugate gradient methods, it is important to use current information about the problem and the
 * algorithm to make the initial guess. For Newton and quasi-Newton methods, the UNIT step size initialization method
 * should always be selected. This choice ensures that unit step lengths are taken whenever they satisfy the termination
 * conditions and allows the rapid rate of convergence properties of these methods to take effect.
 *
 * @author Emmanouil Antonios Platanios
 */
class StepSizeInitialization {
    /**
     * Computes a value for the initial step size (used by iterative line search algorithms) by assuming that the first
     * order change in the objective function at the current iterate/point will be the same as the one obtained in the
     * previous step.
     *
     * @param   objectiveGradientAtCurrentPoint     The objective function gradient evaluated at the current
     *                                              iterate/point.
     * @param   currentDirection                    The current direction selected by the optimization algorithm.
     * @param   objectiveGradientAtPreviousPoint    The objective function gradient evaluated at the previous
     *                                              iterate/point.
     * @param   previousDirection                   The direction used by the optimization algorithm in the previous
     *                                              iteration.
     * @param   previousStepSize                    The step size used by the optimization algorithm in the previous
     *                                              iterations.
     * @return                                      A value for the initial step size, to be used by iterative line
     *                                              search algorithms.
     */
    public static double computeByConservingFirstOrderChange(RealVector objectiveGradientAtCurrentPoint,
                                                             RealVector currentDirection,
                                                             RealVector objectiveGradientAtPreviousPoint,
                                                             RealVector previousDirection,
                                                             double previousStepSize) {
        return previousStepSize
                * objectiveGradientAtPreviousPoint.dotProduct(previousDirection)
                /  objectiveGradientAtCurrentPoint.dotProduct(currentDirection);
    }

    /**
     * Computes a value for the initial step size (used by iterative line search algorithms) by setting it to equal to
     * the minimizer of a quadratic interpolation to the current data: the objective function value at the current
     * iterate/point, the objective function value at the previous iterate/point, the objective function gradient at the
     * previous iterate/point and the previous direction used by the algorithm.
     *
     * @param   objectiveValueAtCurrentPoint        The objective function value at the current iterate/point.
     * @param   objectiveValueAtPreviousPoint       The objective function value at the previous iterate/point.
     * @param   objectiveGradientAtPreviousPoint    The objective function gradient evaluated at the previous
     *                                              iterate/point.
     * @param   previousDirection                   The direction used by the optimization algorithm in the previous
     *                                              iteration.
     * @return                                      A value for the initial step size, to be used by iterative line
     *                                              search algorithms.
     */
    public static double computeByQuadraticInterpolation(double objectiveValueAtCurrentPoint,
                                                         double objectiveValueAtPreviousPoint,
                                                         RealVector objectiveGradientAtPreviousPoint,
                                                         RealVector previousDirection) {
        return 2 * (objectiveValueAtCurrentPoint - objectiveValueAtPreviousPoint)
                / objectiveGradientAtPreviousPoint.dotProduct(previousDirection);
    }

    /**
     * Computes a value for the initial step size (used by iterative line search algorithms) by setting it to equal to
     * the minimum between 1 and 1.01 times the minimizer of a quadratic interpolation to the current data: the
     * objective function value at the current iterate/point, the objective function value at the previous
     * iterate/point, the objective function gradient at the previous iterate/point and the previous direction used by
     * the algorithm.
     *
     * @param   objectiveValueAtCurrentPoint        The objective function value at the current iterate/point.
     * @param   objectiveValueAtPreviousPoint       The objective function value at the previous iterate/point.
     * @param   objectiveGradientAtPreviousPoint    The objective function gradient evaluated at the previous
     *                                              iterate/point.
     * @param   previousDirection                   The direction used by the optimization algorithm in the previous
     *                                              iteration.
     * @return                                      A value for the initial step size, to be used by iterative line
     *                                              search algorithms.
     */
    public static double computeByModifiedQuadraticInterpolation(double objectiveValueAtCurrentPoint,
                                                                 double objectiveValueAtPreviousPoint,
                                                                 RealVector objectiveGradientAtPreviousPoint,
                                                                 RealVector previousDirection) {
        return Math.min(1, 2.02 * (objectiveValueAtCurrentPoint - objectiveValueAtPreviousPoint)
                / objectiveGradientAtPreviousPoint.dotProduct(previousDirection));
    }
}
