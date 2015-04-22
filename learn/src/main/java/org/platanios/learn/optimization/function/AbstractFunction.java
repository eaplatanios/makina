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
     * Computes the function value at a particular point.
     *
     * @param   point   The point in which to evaluate the function.
     * @return          The value of the function, evaluated at the given point.
     */
    public final double getValue(Vector point) {
        numberOfFunctionEvaluations++;
        return computeValue(point);
    }

    abstract protected double computeValue(Vector point);

    /**
     * Computes the first derivatives of the function at a particular point.
     *
     * @param   point   The point in which to evaluate the derivatives.
     * @return          The values of the first derivatives of the function, evaluated at the given point.
     */
    public final Vector getGradient(Vector point) throws NonSmoothFunctionException {
        numberOfGradientEvaluations++;
        return computeGradient(point);
    }

    protected Vector computeGradient(Vector point) throws NonSmoothFunctionException {
        if (computeGradientMethodOverridden)
            computeGradientMethodOverridden = false;
        return derivativesApproximation.approximateGradient(point);
    }

    /**
     * Computes the Hessian of the function at a particular point.
     *
     * @param   point   The point in which to evaluate the Hessian.
     * @return          The value of the Hessian matrix of the function, evaluated at the given point.
     */
    public final Matrix getHessian(Vector point) throws NonSmoothFunctionException {
        numberOfHessianEvaluations++;
        return computeHessian(point);
    }

    protected Matrix computeHessian(Vector point) throws NonSmoothFunctionException {
        if (computeGradientMethodOverridden)
            return derivativesApproximation.approximateHessianGivenGradient(point);
        else
            return derivativesApproximation.approximateHessian(point);
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
