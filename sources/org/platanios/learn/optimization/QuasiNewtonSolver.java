package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.NonPositiveDefiniteMatrixException;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Emmanouil Antonios Platanios
 */
public class QuasiNewtonSolver extends AbstractLineSearchSolver {
    private final Matrix identityMatrix;

    private Method method = Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO;
    private Matrix currentH;
    private Matrix previousH;
    Vector[] s;
    Vector[] y;
    private int m;
    private Vector initialHessianInverseDiagonal = new Vector(currentPoint.getDimension(), 1);

    private double symmetricRankOneSkippingParameter = 1e-8;

    public static class Builder {
        // Required parameters
        private final AbstractFunction objective;
        private final double[] initialPoint;

        // Optional parameters - Initialized to default values
        private Method method = Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO;
        private int m = 1;
        private double symmetricRankOneSkippingParameter = 1e-8;

        public Builder(AbstractFunction objective, double[] initialPoint) {
            this.objective = objective;
            this.initialPoint = initialPoint;
        }

        public Builder method(Method method) {
            this.method = method;

            if (method != Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO) {
                m = 1;
            } else if (m == 1) {
                m = 10;
            }

            return this;
        }

        public Builder m(int m) {
            if (method != Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO) {
                m = 1;
            }

            this.m = m;
            return this;
        }

        public Builder symmetricRankOneSkippingParameter(double symmetricRankOneSkippingParameter) {
            this.symmetricRankOneSkippingParameter = symmetricRankOneSkippingParameter;
            return this;
        }

        public QuasiNewtonSolver build() {
            return new QuasiNewtonSolver(this);
        }
    }

    /**
     * BROYDEN_FLETCHER_GOLDFARB_SHANNO method chosen by default. The default value used for the memory parameter,
     * {@code m}, is 10 for L-BFGS (for all other methods it is 1, since they store the whole approximation matrix and
     * do not need to store any previous vectors to re-construct it using limited memory).
     *
     */
    public QuasiNewtonSolver(Builder builder) {
        super(builder.objective, builder.initialPoint);
        this.method = builder.method;
        this.m = builder.m;
        this.symmetricRankOneSkippingParameter = builder.symmetricRankOneSkippingParameter;
        StrongWolfeInterpolationLineSearch lineSearch = new StrongWolfeInterpolationLineSearch(objective,
                                                                                               1e-4,
                                                                                               0.9,
                                                                                               1000);
        lineSearch.setStepSizeInitializationMethod(StepSizeInitializationMethod.UNIT);
        setLineSearch(lineSearch);
        identityMatrix = Matrix.generateIdentityMatrix(builder.initialPoint.length);
        currentH = identityMatrix;
        currentGradient = objective.getGradient(currentPoint);
        s = new Vector[m];
        y = new Vector[m];
    }

    @Override
    public void updateDirection() {
        switch (method) {
            case DAVIDON_FLETCHER_POWELL:
            case BROYDEN_FLETCHER_GOLDFARB_SHANNO:
            case SYMMETRIC_RANK_ONE:
            case BROYDEN:
                if (currentIteration > 0) {
                    // Simple trick to initialize the inverse Hessian matrix approximation. This scaling tries to make
                    // the size of H similar to the size of the actual Hessian matrix inverse (the scaling factor
                    // attempts to estimate the size of the true Hessian matrix along the most recent search direction.
                    // This choice helps to ensure that the search direction is well scaled and as a result the step
                    // length value 1 is accepted in most iterations.
                    if (currentIteration == 1) {
                        previousH = currentH.multiply(y[0].innerProduct(s[0]) / y[0].innerProduct(y[0]));
                    } else {
                        previousH = currentH;
                    }
                    updateHessianInverseApproximation();
                }
                currentDirection = currentH.multiply(currentGradient).multiply(-1);
                break;
            case LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO:
                // Same trick to initialize the inverse Hessian matrix approximation as that used above for the other
                // methods.
                if (currentIteration > 0) {
                    initialHessianInverseDiagonal =
                            (new Vector(currentPoint.getDimension(), 1))
                                    .multiply(s[0].innerProduct(y[0]) / y[0].innerProduct(y[0]));
                }
                currentDirection = approximateHessianInverseVectorProduct(currentGradient).multiply(-1);
                break;
        }
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.multiply(currentStepSize));
        currentGradient = objective.getGradient(currentPoint);
        updateStoredVectors();
    }

    /** Used from all methods apart from the LBFGS method. */
    private void updateHessianInverseApproximation() {
        double rho = 1 / y[0].innerProduct(s[0]);

        switch (method) {
            case DAVIDON_FLETCHER_POWELL:
                currentH = previousH.subtract(previousH.multiply(
                        y[0].multiply(1 / y[0].multiply(previousH).innerProduct(y[0]))
                                .outerProduct(y[0])
                                .multiply(previousH)
                )).add(s[0].multiply(rho).outerProduct(s[0]));
                break;
            case BROYDEN_FLETCHER_GOLDFARB_SHANNO:
                currentH = identityMatrix.subtract(s[0].multiply(rho).outerProduct(y[0]))
                        .multiply(previousH)
                        .multiply(identityMatrix.subtract(y[0].multiply(rho).outerProduct(s[0])))
                        .add(s[0].multiply(rho).outerProduct(s[0]));
                break;
            case SYMMETRIC_RANK_ONE:
                Vector tempVector = s[0].subtract(previousH.multiply(y[0]));
                if (Math.abs(tempVector.innerProduct(y[0]))
                        >= symmetricRankOneSkippingParameter * y[0].computeL2Norm() * tempVector.computeL2Norm()) {
                    currentH = previousH.add(
                            tempVector.multiply(1 / tempVector.innerProduct(y[0])).outerProduct(tempVector)
                    );
                } else {
                    currentH = previousH;
                }
                break;
            case BROYDEN:
                currentH = previousH.add(
                        s[0].subtract(previousH.multiply(y[0]))
                                .outerProduct(s[0].multiply(previousH))
                                .multiply(1 / s[0].multiply(previousH).innerProduct(y[0]))
                );
                break;
            default:
                throw new NotImplementedException();
        }
    }

    /** Used only for the LBFGS method.
     * @param q*/
    private Vector approximateHessianInverseVectorProduct(Vector q) {
        double[] a = new double[m];
        double[] rho = new double[m];
        for (int i = 0; i < Math.min(m, currentIteration); i++) {
            rho[i] = 1 / y[i].innerProduct(s[i]);
            a[i] = rho[i] * s[i].innerProduct(q);
            q = q.subtract(y[i].multiply(a[i]));
        }
        Vector result = q.multiplyElementwise(initialHessianInverseDiagonal);
        for (int i = Math.min(m, currentIteration) - 1; i >= 0; i--) {
            result = result.add(s[i].multiply(a[i] - rho[i] * y[i].innerProduct(result)));
        }
        return result;
    }

    /** m != 1 only for the LBFGS method. */
    private void updateStoredVectors() {
        for (int i = m - 1; i > 0; i--) {
            s[i] = s[i-1];
            y[i] = y[i-1];
        }
        s[0] = currentPoint.subtract(previousPoint);
        y[0] = currentGradient.subtract(previousGradient);
    }

    public enum Method {
        /** The Davidon–Fletcher–Powell algorithm. This algorithm is less effective than BROYDEN_FLETCHER_GOLDFARB_SHANNO at correcting bad Hessian
         * approximations. Both this method and the BROYDEN_FLETCHER_GOLDFARB_SHANNO method preserve the positive-definiteness of the Hessian matrix. */
        DAVIDON_FLETCHER_POWELL,
        /** The Broyden–Fletcher–Goldfarb–Shanno algorithm. This algorithm is very good at correcting bad Hessian
         * approximations. */
        BROYDEN_FLETCHER_GOLDFARB_SHANNO,
        LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO,
        /** The Symmetric-Rank-1 algorithm. This method may produce indefinite Hessian approximations. Furthermore, the
         * basic SYMMETRIC_RANK_ONE method may break down and that is why here it has been implemented with a skipping method to help
         * prevent such cases. */
        SYMMETRIC_RANK_ONE,
        BROYDEN

    }
}
