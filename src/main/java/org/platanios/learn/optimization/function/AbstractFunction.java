package org.platanios.learn.optimization.function;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractFunction {
    private int numberOfFunctionEvaluations = 0;
    private int numberOfGradientEvaluations = 0;
    private int numberOfHessianEvaluations = 0;
    private boolean computeGradientMethodOverridden = true;

    private DerivativesApproximation derivativesApproximation =
            new DerivativesApproximation(this, DerivativesApproximation.Method.CENTRAL_DIFFERENCE);

    /**
     * Computes the objective function value and the constraints values at a particular point.
     *
     * @param   point   The point in which to evaluate the objective function and the constraints.
     * @return          The value of the objective function, evaluated at the given point.
     */
    public final double getValue(Vector point) {
        numberOfFunctionEvaluations++;
        return computeValue(point);
    }

    abstract public double computeValue(Vector point);

    /**
     * Computes the first derivatives of the objective function and the constraints at a particular point.
     *
     * @param   point   The point in which to evaluate the derivatives.
     * @return          The values of the first derivatives of the objective function, evaluated at the given point.
     */
    public final Vector getGradient(Vector point) {
        numberOfGradientEvaluations++;
        return computeGradient(point);
    }

    public Vector computeGradient(Vector point) {
        if (computeGradientMethodOverridden) {
            computeGradientMethodOverridden = false;
        }

        return derivativesApproximation.approximateGradient(point);
    }

    /**
     * Computes the Hessian of the objective function at a particular point.
     *
     * @param   point   The point in which to evaluate the Hessian.
     * @return          The value of the Hessian matrix of the objective function, evaluated at the given point.
     */
    public final Matrix getHessian(Vector point) {
        numberOfHessianEvaluations++;
        return computeHessian(point);
    }

    public Matrix computeHessian(Vector point) {
        if (computeGradientMethodOverridden) {
            return derivativesApproximation.approximateHessianGivenGradient(point);
        } else {
            return derivativesApproximation.approximateHessian(point);
        }
    }

    public final int getNumberOfFunctionEvaluations() {
        return numberOfFunctionEvaluations;
    }

    public final int getNumberOfGradientEvaluations() {
        return numberOfGradientEvaluations;
    }

    public final int getNumberOfHessianEvaluations() {
        return numberOfHessianEvaluations;
    }

    public final DerivativesApproximation.Method getDerivativesApproximationMethod() {
        return derivativesApproximation.getMethod();
    }

    public final void setDerivativesApproximationMethod(DerivativesApproximation.Method method) {
        derivativesApproximation.setMethod(method);
    }
}
