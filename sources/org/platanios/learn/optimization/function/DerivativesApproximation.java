package org.platanios.learn.optimization.function;

import org.platanios.learn.math.Utilities;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * TODO: Jacobian approximation.
 *
 * @author Emmanouil Antonios Platanios
 */
public class DerivativesApproximation {
    private final AbstractFunction function;
    private final double epsilon;

    private DerivativesApproximationMethod method;

    public DerivativesApproximation(AbstractFunction function, DerivativesApproximationMethod method) {
        this.function = function;
        this.method = method;

        switch (method) {
            case FORWARD_DIFFERENCE:
                epsilon = Math.sqrt(Utilities.computeMachineEpsilonDouble());
                break;
            case CENTRAL_DIFFERENCE:
                epsilon = Math.cbrt(Utilities.computeMachineEpsilonDouble());
                break;
            default:
                throw new NotImplementedException();
        }
    }

    public org.platanios.learn.math.matrix.Vector approximateGradient(Vector point) {
        int n = point.getDimension();
        Vector gradient = new Vector(n, 0);
        Vector ei = new Vector(n, 0);

        switch(method) {
            case FORWARD_DIFFERENCE:
                double currentFunctionValue = function.getValue(point);
                for (int i = 0; i < n; i++) {
                    ei.setAllElements(0);
                    ei.setElement(i, 1);
                    double forwardFunctionValue = function.getValue(point.add(ei.multiply(epsilon)));
                    gradient.setElement(i, (forwardFunctionValue - currentFunctionValue) / epsilon);
                }
            break;
            case CENTRAL_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.setAllElements(0);
                    ei.setElement(i, 1);
                    double forwardFunctionValue = function.getValue(point.add(ei.multiply(epsilon)));
                    double backwardFunctionValue = function.getValue(point.subtract(ei.multiply(epsilon)));
                    gradient.setElement(i, (forwardFunctionValue - backwardFunctionValue) / (2 * epsilon));
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return gradient;
    }

    public Matrix approximateHessian(Vector point) {
        int n = point.getDimension();
        Matrix hessian = new Matrix(n, n);
        Vector ei = new Vector(n, 0);
        Vector ej = new Vector(n, 0);
        double currentFunctionValue = function.getValue(point);

        switch(method) {
            case FORWARD_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.setAllElements(0);
                    ei.setElement(i, 1);
                    double iFunctionValue = function.getValue(point.add(ei.multiply(epsilon)));
                    for (int j = i; j < n; j++) {
                        ej.setAllElements(0);
                        ej.setElement(j, 1);
                        double jFunctionValue = function.getValue(point.add(ej.multiply(epsilon)));
                        double ijFunctionValue = function.getValue(point.add(ei.add(ej).multiply(epsilon)));
                        double ijEntry = (ijFunctionValue - iFunctionValue - jFunctionValue + currentFunctionValue)
                                / Math.pow(epsilon, 2);
                        hessian.setElement(i, j, ijEntry);
                        if (i != j) {
                            hessian.setElement(j, i, ijEntry);
                        }
                    }
                }
                break;
            case CENTRAL_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.setAllElements(0);
                    ei.setElement(i, 1);
                    for (int j = i; j < n; j++) {
                        ej.setAllElements(0);
                        ej.setElement(j, 1);
                        if (i != j) {
                            double term1 = function.getValue(point.add(ei.add(ej).multiply(epsilon)));
                            double term2 = function.getValue(point.add(ei.subtract(ej).multiply(epsilon)));
                            double term3 = function.getValue(point.add(ei.subtract(ej).multiply(-epsilon)));
                            double term4 = function.getValue(point.add(ei.add(ej).multiply(-epsilon)));
                            double ijEntry = (term1 - term2 - term3 + term4) / (4 * Math.pow(epsilon, 2));
                            hessian.setElement(i, j, ijEntry);
                            hessian.setElement(j, i, ijEntry);
                        } else {
                            double term1 = function.getValue(point.add(ei.multiply(2 * epsilon)));
                            double term2 = function.getValue(point.add(ei.multiply(epsilon)));
                            double term3 = function.getValue(point.subtract(ei.multiply(epsilon)));
                            double term4 = function.getValue(point.subtract(ei.multiply(2 * epsilon)));
                            double ijEntry = (- term1 + 16 * term2 - 30 * currentFunctionValue + 16 * term3 - term4)
                                    / (12 * Math.pow(epsilon, 2));
                            hessian.setElement(i, j, ijEntry);
                        }
                    }
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return hessian;
    }

    public Matrix approximateHessianGivenGradient(Vector point) {
        int n = point.getDimension();
        Matrix hessian = new Matrix(n, n);
        Vector ei = new Vector(n, 0);

        switch(method) {
            case FORWARD_DIFFERENCE:
                Vector currentGradientValue = function.getGradient(point);
                for (int i = 0; i < n; i++) {
                    ei.setAllElements(0);
                    ei.setElement(i, 1);
                    Vector forwardGradientValue = function.getGradient(point.add(ei.multiply(epsilon)));
                    hessian.setSubMatrix(
                            0,
                            hessian.getRowDimension() - 1,
                            i,
                            i,
                            forwardGradientValue.subtract(currentGradientValue).multiply(1 / epsilon).copyAsMatrix()
                    );
                }
                break;
            case CENTRAL_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.setAllElements(0);
                    ei.setElement(i, 1);
                    Vector forwardGradientValue = function.getGradient(point.add(ei.multiply(epsilon)));
                    Vector backwardGradientValue = function.getGradient(point.subtract(ei.multiply(epsilon)));
                    hessian.setSubMatrix(
                            0,
                            hessian.getRowDimension() - 1,
                            i,
                            i,
                            forwardGradientValue
                                    .subtract(backwardGradientValue).multiply(1 / (2 * epsilon)).copyAsMatrix()
                    );
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return hessian;
    }

    public Vector approximateHessianVectorProductGivenGradient(Vector point, Vector p) {
        Vector result;
        Vector forwardGradientValue = function.getGradient(point.add(p.multiply(epsilon)));

        switch(method) {
            case FORWARD_DIFFERENCE:
                Vector currentGradientValue = function.getGradient(point);
                result = forwardGradientValue.subtract(currentGradientValue).multiply(1 / epsilon);
                break;
            case CENTRAL_DIFFERENCE:
                Vector backwardGradientValue = function.getGradient(point.subtract(p.multiply(epsilon)));
                result = forwardGradientValue.subtract(backwardGradientValue).multiply(1 / (2 * epsilon));
                break;
            default:
                throw new NotImplementedException();
        }

        return result;
    }

    public DerivativesApproximationMethod getMethod() {
        return method;
    }

    public void setMethod(DerivativesApproximationMethod method) {
        this.method = method;
    }
}
