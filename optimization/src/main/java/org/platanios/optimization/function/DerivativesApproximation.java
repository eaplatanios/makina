package org.platanios.optimization.function;

import org.platanios.math.MathUtilities;
import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;

/**
 * TODO: Jacobian approximation.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class DerivativesApproximation {
    private final AbstractFunction function;
    private final double epsilon;

    private Method method;

    public DerivativesApproximation(AbstractFunction function, Method method) {
        this(function, method, method.computeEpsilon());
    }

    public DerivativesApproximation(AbstractFunction function, Method method, double epsilon) {
        this.function = function;
        this.method = method;
        this.epsilon = epsilon;
    }

    public Vector approximateGradient(Vector point) {
        return method.approximateGradient(this, point);
    }

    public Matrix approximateHessian(Vector point) {
        return method.approximateHessian(this, point);
    }

    public Matrix approximateHessianGivenGradient(Vector point)
            throws NonSmoothFunctionException {
        return method.approximateHessianGivenGradient(this, point);
    }

    public Vector approximateHessianVectorProductGivenGradient(Vector point, Vector p)
            throws NonSmoothFunctionException {
        return method.approximateHessianVectorProductGivenGradient(this, point, p);
    }

    public Method getMethod() {
        return method;
    }

    public void setMethod(Method method) {
        this.method = method;
    }

    public enum Method {
        FORWARD_DIFFERENCE {
            protected double computeEpsilon() {
                return Math.sqrt(MathUtilities.computeMachineEpsilonDouble());
            }

            protected Vector approximateGradient(DerivativesApproximation owner, Vector point) {
                int n = point.size();
                Vector gradient = Vectors.dense(n, 0);
                Vector ei = Vectors.dense(n, 0);
                double currentFunctionValue = owner.function.getValue(point);
                for (int i = 0; i < n; i++) {
                    ei.setAll(0);
                    ei.set(i, 1);
                    double forwardFunctionValue = owner.function.getValue(point.add(ei.mult(owner.epsilon)));
                    gradient.set(i, (forwardFunctionValue - currentFunctionValue) / owner.epsilon);
                }
                return gradient;
            }

            protected Matrix approximateHessian(DerivativesApproximation owner, Vector point) {
                int n = point.size();
                Matrix hessian = new Matrix(n, n);
                Vector ei = Vectors.dense(n, 0);
                Vector ej = Vectors.dense(n, 0);
                double currentFunctionValue = owner.function.getValue(point);
                for (int i = 0; i < n; i++) {
                    ei.setAll(0);
                    ei.set(i, 1);
                    double iFunctionValue = owner.function.getValue(point.add(ei.mult(owner.epsilon)));
                    for (int j = i; j < n; j++) {
                        ej.setAll(0);
                        ej.set(j, 1);
                        double jFunctionValue = owner.function.getValue(point.add(ej.mult(owner.epsilon)));
                        double ijFunctionValue = owner.function.getValue(point.add(ei.add(ej).mult(owner.epsilon)));
                        double ijEntry = (ijFunctionValue - iFunctionValue - jFunctionValue + currentFunctionValue)
                                / Math.pow(owner.epsilon, 2);
                        hessian.setElement(i, j, ijEntry);
                        if (i != j) {
                            hessian.setElement(j, i, ijEntry);
                        }
                    }
                }
                return hessian;
            }

            protected Matrix approximateHessianGivenGradient(DerivativesApproximation owner, Vector point)
                    throws NonSmoothFunctionException {
                int n = point.size();
                Matrix hessian = new Matrix(n, n);
                Vector ei = Vectors.dense(n, 0);
                Vector currentGradientValue = owner.function.getGradient(point);
                for (int i = 0; i < n; i++) {
                    ei.setAll(0);
                    ei.set(i, 1);
                    Vector forwardGradientValue = owner.function.getGradient(point.add(ei.mult(owner.epsilon)));
                    hessian.setColumn(
                            i,
                            forwardGradientValue.sub(currentGradientValue).mult(1 / owner.epsilon)
                    );
                }
                return hessian;
            }

            protected Vector approximateHessianVectorProductGivenGradient(DerivativesApproximation owner,
                                                                          Vector point,
                                                                          Vector p)
                    throws NonSmoothFunctionException {
                Vector forwardGradientValue = owner.function.getGradient(point.add(p.mult(owner.epsilon)));
                Vector currentGradientValue = owner.function.getGradient(point);
                return forwardGradientValue.sub(currentGradientValue).mult(1 / owner.epsilon);
            }
        },
        /** Much more accurate than the forward-difference method (\(O(\epsilon^2)\) estimation error instead of
         * \(O(\epsilon)\)). */
        CENTRAL_DIFFERENCE {
            protected double computeEpsilon() {
                return Math.cbrt(MathUtilities.computeMachineEpsilonDouble());
            }

            protected Vector approximateGradient(DerivativesApproximation owner, Vector point) {
                int n = point.size();
                Vector gradient = Vectors.dense(n, 0);
                Vector ei = Vectors.dense(n, 0);
                for (int i = 0; i < n; i++) {
                    ei.setAll(0);
                    ei.set(i, 1);
                    double forwardFunctionValue = owner.function.getValue(point.add(ei.mult(owner.epsilon)));
                    double backwardFunctionValue = owner.function.getValue(point.sub(ei.mult(owner.epsilon)));
                    gradient.set(i, (forwardFunctionValue - backwardFunctionValue) / (2 * owner.epsilon));
                }
                return gradient;
            }

            protected Matrix approximateHessian(DerivativesApproximation owner, Vector point) {
                int n = point.size();
                Matrix hessian = new Matrix(n, n);
                Vector ei = Vectors.dense(n, 0);
                Vector ej = Vectors.dense(n, 0);
                double currentFunctionValue = owner.function.getValue(point);
                for (int i = 0; i < n; i++) {
                    ei.setAll(0);
                    ei.set(i, 1);
                    for (int j = i; j < n; j++) {
                        ej.setAll(0);
                        ej.set(j, 1);
                        if (i != j) {
                            double term1 = owner.function.getValue(point.add(ei.add(ej).mult(owner.epsilon)));
                            double term2 = owner.function.getValue(point.add(ei.sub(ej).mult(owner.epsilon)));
                            double term3 = owner.function.getValue(point.add(ei.sub(ej).mult(-owner.epsilon)));
                            double term4 = owner.function.getValue(point.add(ei.add(ej).mult(-owner.epsilon)));
                            double ijEntry = (term1 - term2 - term3 + term4) / (4 * Math.pow(owner.epsilon, 2));
                            hessian.setElement(i, j, ijEntry);
                            hessian.setElement(j, i, ijEntry);
                        } else {
                            double term1 = owner.function.getValue(point.add(ei.mult(2 * owner.epsilon)));
                            double term2 = owner.function.getValue(point.add(ei.mult(owner.epsilon)));
                            double term3 = owner.function.getValue(point.sub(ei.mult(owner.epsilon)));
                            double term4 = owner.function.getValue(point.sub(ei.mult(2 * owner.epsilon)));
                            double ijEntry = (- term1 + 16 * term2 - 30 * currentFunctionValue + 16 * term3 - term4)
                                    / (12 * Math.pow(owner.epsilon, 2));
                            hessian.setElement(i, j, ijEntry);
                        }
                    }
                }
                return hessian;
            }

            protected Matrix approximateHessianGivenGradient(DerivativesApproximation owner, Vector point)
                    throws NonSmoothFunctionException {
                int n = point.size();
                Matrix hessian = new Matrix(n, n);
                Vector ei = Vectors.dense(n, 0);
                for (int i = 0; i < n; i++) {
                    ei.setAll(0);
                    ei.set(i, 1);
                    Vector forwardGradientValue = owner.function.getGradient(point.add(ei.mult(owner.epsilon)));
                    Vector backwardGradientValue =
                            owner.function.getGradient(point.sub(ei.mult(owner.epsilon)));
                    hessian.setColumn(
                            i,
                            forwardGradientValue.sub(backwardGradientValue).mult(1 / (2 * owner.epsilon))
                    );
                }
                return hessian;
            }

            protected Vector approximateHessianVectorProductGivenGradient(DerivativesApproximation owner,
                                                                          Vector point,
                                                                          Vector p)
                    throws NonSmoothFunctionException {
                Vector forwardGradientValue = owner.function.getGradient(point.add(p.mult(owner.epsilon)));
                Vector backwardGradientValue = owner.function.getGradient(point.sub(p.mult(owner.epsilon)));
                return forwardGradientValue.sub(backwardGradientValue).mult(1 / (2 * owner.epsilon));
            }
        };

        protected abstract double computeEpsilon();
        protected abstract Vector approximateGradient(DerivativesApproximation owner, Vector point);
        protected abstract Matrix approximateHessian(DerivativesApproximation owner, Vector point);
        protected abstract Matrix approximateHessianGivenGradient(DerivativesApproximation owner, Vector point) throws NonSmoothFunctionException;
        protected abstract Vector approximateHessianVectorProductGivenGradient(DerivativesApproximation owner,
                                                                               Vector point,
                                                                               Vector p) throws NonSmoothFunctionException;
    }
}
