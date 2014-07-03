package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.math.Utilities;
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
                epsilon = Math.sqrt(Utilities.calculateMachineEpsilonDouble());
                break;
            case CENTRAL_DIFFERENCE:
                epsilon = Math.cbrt(Utilities.calculateMachineEpsilonDouble());
                break;
            default:
                throw new NotImplementedException();
        }
    }

    public RealVector approximateGradient(RealVector point) {
        int n = point.getDimension();
        RealVector gradient = new ArrayRealVector(n, 0);
        RealVector ei = new ArrayRealVector(n, 0);

        switch(method) {
            case FORWARD_DIFFERENCE:
                double currentFunctionValue = function.getValue(point);
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    double forwardFunctionValue = function.getValue(point.add(ei.mapMultiply(epsilon)));
                    gradient.setEntry(i, (forwardFunctionValue - currentFunctionValue) / epsilon);
                }
            break;
            case CENTRAL_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    double forwardFunctionValue = function.getValue(point.add(ei.mapMultiply(epsilon)));
                    double backwardFunctionValue = function.getValue(point.subtract(ei.mapMultiply(epsilon)));
                    gradient.setEntry(i, (forwardFunctionValue - backwardFunctionValue) / (2 * epsilon));
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return gradient;
    }

    public RealMatrix approximateHessian(RealVector point) {
        int n = point.getDimension();
        RealMatrix hessian = new Array2DRowRealMatrix(n, n);
        RealVector ei = new ArrayRealVector(n, 0);
        RealVector ej = new ArrayRealVector(n, 0);
        double currentFunctionValue = function.getValue(point);

        switch(method) {
            case FORWARD_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    double iFunctionValue = function.getValue(point.add(ei.mapMultiply(epsilon)));
                    for (int j = i; j < n; j++) {
                        ej.set(0);
                        ej.setEntry(j, 1);
                        double jFunctionValue = function.getValue(point.add(ej.mapMultiply(epsilon)));
                        double ijFunctionValue = function.getValue(point.add(ei.add(ej).mapMultiply(epsilon)));
                        double ijEntry = (ijFunctionValue - iFunctionValue - jFunctionValue + currentFunctionValue)
                                / Math.pow(epsilon, 2);
                        hessian.setEntry(i, j, ijEntry);
                        if (i != j) {
                            hessian.setEntry(j, i, ijEntry);
                        }
                    }
                }
                break;
            case CENTRAL_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    for (int j = i; j < n; j++) {
                        ej.set(0);
                        ej.setEntry(j, 1);
                        if (i != j) {
                            double term1 = function.getValue(point.add(ei.add(ej).mapMultiply(epsilon)));
                            double term2 = function.getValue(point.add(ei.subtract(ej).mapMultiply(epsilon)));
                            double term3 = function.getValue(point.add(ei.subtract(ej).mapMultiply(-epsilon)));
                            double term4 = function.getValue(point.add(ei.add(ej).mapMultiply(-epsilon)));
                            double ijEntry = (term1 - term2 - term3 + term4) / (4 * Math.pow(epsilon, 2));
                            hessian.setEntry(i, j, ijEntry);
                            hessian.setEntry(j, i, ijEntry);
                        } else {
                            double term1 = function.getValue(point.add(ei.mapMultiply(2 * epsilon)));
                            double term2 = function.getValue(point.add(ei.mapMultiply(epsilon)));
                            double term3 = function.getValue(point.subtract(ei.mapMultiply(epsilon)));
                            double term4 = function.getValue(point.subtract(ei.mapMultiply(2 * epsilon)));
                            double ijEntry = (- term1 + 16 * term2 - 30 * currentFunctionValue + 16 * term3 - term4)
                                    / (12 * Math.pow(epsilon, 2));
                            hessian.setEntry(i, j, ijEntry);
                        }
                    }
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return hessian;
    }

    public RealMatrix approximateHessianGivenGradient(RealVector point) {
        int n = point.getDimension();
        RealMatrix hessian = new Array2DRowRealMatrix(n, n);
        RealVector ei = new ArrayRealVector(n, 0);

        switch(method) {
            case FORWARD_DIFFERENCE:
                RealVector currentGradientValue = function.getGradient(point);
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    RealVector forwardGradientValue = function.getGradient(point.add(ei.mapMultiply(epsilon)));
                    hessian.setColumnVector(
                            i,
                            forwardGradientValue.subtract(currentGradientValue).mapMultiply(1 / epsilon)
                    );
                }
                break;
            case CENTRAL_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    RealVector forwardGradientValue = function.getGradient(point.add(ei.mapMultiply(epsilon)));
                    RealVector backwardGradientValue = function.getGradient(point.subtract(ei.mapMultiply(epsilon)));
                    hessian.setColumnVector(
                            i,
                            forwardGradientValue.subtract(backwardGradientValue).mapMultiply(1 / (2 * epsilon))
                    );
                }
                break;
            default:
                throw new NotImplementedException();
        }

        return hessian;
    }

    public RealVector approximateHessianVectorProductGivenGradient(RealVector point, RealVector p) {
        RealVector result;
        RealVector forwardGradientValue = function.getGradient(point.add(p.mapMultiply(epsilon)));

        switch(method) {
            case FORWARD_DIFFERENCE:
                RealVector currentGradientValue = function.getGradient(point);
                result = forwardGradientValue.subtract(currentGradientValue).mapMultiply(1 / epsilon);
                break;
            case CENTRAL_DIFFERENCE:
                RealVector backwardGradientValue = function.getGradient(point.subtract(p.mapMultiply(epsilon)));
                result = forwardGradientValue.subtract(backwardGradientValue).mapMultiply(1 / (2 * epsilon));
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
