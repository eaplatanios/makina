package org.platanios.learn.optimization.function;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DerivativeApproximation {
    private final Function function;
    private final DerivativeApproximationMethod method;
    private final double epsilon;

    public DerivativeApproximation(Function function, DerivativeApproximationMethod method) {
        this.function = function;
        this.method = method;

        switch (method) {
            case FORWARD_DIFFERENCE:
                epsilon = Math.sqrt(calculateMachineEpsilonDouble());
                break;
            case CENTRAL_DIFFERENCE:
                epsilon = Math.cbrt(calculateMachineEpsilonDouble());
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
                double currentFunctionValue = function.computeValue(point);
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    double forwardFunctionValue = function.computeValue(point.add(ei.mapMultiply(epsilon)));
                    gradient.setEntry(i, (forwardFunctionValue - currentFunctionValue) / epsilon);
                }
            break;
            case CENTRAL_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    double forwardFunctionValue = function.computeValue(point.add(ei.mapMultiply(epsilon)));
                    double backwardFunctionValue = function.computeValue(point.subtract(ei.mapMultiply(epsilon)));
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
        double currentFunctionValue = function.computeValue(point);

        switch(method) {
            case FORWARD_DIFFERENCE:
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    double iFunctionValue = function.computeValue(point.add(ei.mapMultiply(epsilon)));
                    for (int j = i; j < n; j++) {
                        ej.set(0);
                        ej.setEntry(j, 1);
                        double jFunctionValue = function.computeValue(point.add(ej.mapMultiply(epsilon)));
                        double ijFunctionValue = function.computeValue(point.add(ei.add(ej).mapMultiply(epsilon)));
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
                            double term1 = function.computeValue(point.add(ei.add(ej).mapMultiply(epsilon)));
                            double term2 = function.computeValue(point.add(ei.subtract(ej).mapMultiply(epsilon)));
                            double term3 = function.computeValue(point.add(ei.subtract(ej).mapMultiply(-epsilon)));
                            double term4 = function.computeValue(point.add(ei.add(ej).mapMultiply(-epsilon)));
                            double ijEntry = (term1 - term2 - term3 + term4) / (4 * Math.pow(epsilon, 2));
                            hessian.setEntry(i, j, ijEntry);
                            hessian.setEntry(j, i, ijEntry);
                        } else {
                            double term1 = function.computeValue(point.add(ei.mapMultiply(2 * epsilon)));
                            double term2 = function.computeValue(point.add(ei.mapMultiply(epsilon)));
                            double term3 = function.computeValue(point.subtract(ei.mapMultiply(epsilon)));
                            double term4 = function.computeValue(point.subtract(ei.mapMultiply(2 * epsilon)));
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
                RealVector currentGradientValue = function.computeGradient(point);
                for (int i = 0; i < n; i++) {
                    ei.set(0);
                    ei.setEntry(i, 1);
                    RealVector forwardGradientValue = function.computeGradient(point.add(ei.mapMultiply(epsilon)));
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
                    RealVector forwardGradientValue = function.computeGradient(point.add(ei.mapMultiply(epsilon)));
                    RealVector backwardGradientValue = function.computeGradient(point.subtract(ei.mapMultiply(epsilon)));
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

    private static double calculateMachineEpsilonDouble() {
        double epsilon = 1;
        while (1 + epsilon / 2 > 1.0) {
            epsilon /= 2;
        }
        return epsilon;
    }
}
