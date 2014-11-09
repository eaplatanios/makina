package org.platanios.learn.classification.reflection.perception;

/**
 * An interface specifying the methods that all classes defined as possible objective functions for the numerical
 * optimization problem involved in the error rates estimation process should implement.
 *
 * @author Emmanouil Antonios Platanios
 */
interface ObjectiveFunction {
    /**
     * Computes the objective value at a particular point. The result is stored in the array passed an argument to this
     * method.
     *
     * @param   point           The point in which to evaluate the objective function.
     * @param   objectiveValue  The array holding objective value to modify.
     */
    void computeObjective(double[] point, double[] objectiveValue);

    /**
     * Computes the first derivatives of the objective function at a particular point. The result is stored in the array
     * passed an argument to this method.
     *
     * @param   point               The point in which to evaluate the derivatives.
     * @param   objectiveGradient   The array holding the objective function gradients to modify.
     */
    void computeGradient(double[] point, double[] objectiveGradient);

    /**
     * Computes the Hessian of the objective function at a particular point. The result is stored in the array passed an
     * argument to this method.
     *
     * @param   point               The point in which to evaluate the Hessian.
     * @param   objectiveHessian    The Hessian (in sparse form) to modify.
     */
    void computeHessian(double[] point, double[] objectiveHessian);
}
