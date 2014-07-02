package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.*;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Emmanouil Antonios Platanios
 */
public class QuasiNewtonSolver extends AbstractLineSearchSolver {
    private final RealMatrix identityMatrix;

    private Method method = Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO;
    private RealMatrix currentH;
    private RealMatrix previousH;
    RealVector[] s;
    RealVector[] y;
    private int m;
    private RealVector initialHessianInverseDiagonal = new ArrayRealVector(currentPoint.getDimension(), 1);

    private double symmetricRankOneSkippingParameter = 1e-8;

    /**
     * BROYDEN_FLETCHER_GOLDFARB_SHANNO method chosen by default. The default value used for the memory parameter,
     * {@code m}, is 10 for L-BFGS (for all other methods it is 1, since they store the whole approximation matrix and
     * do not need to store any previous vectors to re-construct it using limited memory).
     *
     * @param objective
     * @param initialPoint
     */
    public QuasiNewtonSolver(AbstractFunction objective,
                             double[] initialPoint) {
        super(objective, initialPoint);
        StrongWolfeInterpolationLineSearch lineSearch = new StrongWolfeInterpolationLineSearch(objective,
                                                                                               1e-4,
                                                                                               0.9,
                                                                                               1000);
        lineSearch.setStepSizeInitializationMethod(StepSizeInitializationMethod.UNIT);
        setLineSearch(lineSearch);
        identityMatrix = MatrixUtils.createRealIdentityMatrix(initialPoint.length);
        currentH = identityMatrix;
        currentGradient = objective.getGradient(currentPoint);

        if (method != Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO) {
            m = 1;
        } else {
            m = 10;
        }
        s = new RealVector[m];
        y = new RealVector[m];
    }

    @Override
    public void updateDirection() {
        if (method != Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO) {
            if (currentIteration > 0) {
                // Simple trick to initialize the inverse Hessian matrix approximation. This scaling tries to make the
                // size of H similar to the size of the actual Hessian matrix inverse (the scaling factor attempts to
                // estimate the size of the true Hessian matrix along the most recent search direction. This choice
                // helps to ensure that the search direction is well scaled and as a result the step length value 1 is
                // accepted in most iterations.
                if (currentIteration == 1) {
                    previousH = currentH.scalarMultiply(y[0].dotProduct(s[0]) / y[0].dotProduct(y[0]));
                } else {
                    previousH = currentH;
                }
                updateHessianInverseApproximation();
            }
            currentDirection = currentH.operate(currentGradient).mapMultiply(-1);
        } else {
            // Same trick to initialize the inverse Hessian matrix approximation as that used above for the other
            // methods.
            if (currentIteration > 0) {
                initialHessianInverseDiagonal =
                        (new ArrayRealVector(currentPoint.getDimension(), 1))
                                .mapMultiply(s[0].dotProduct(y[0]) / y[0].dotProduct(y[0]));
            }
            currentDirection = approximateHessianInverseVectorProduct(currentGradient).mapMultiply(-1);
        }
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.mapMultiply(currentStepSize));
        currentGradient = objective.getGradient(currentPoint);
        updateStoredVectors();
    }

    /** Used from all methods apart from the LBFGS method. */
    private void updateHessianInverseApproximation() {
        double rho = 1 / y[0].dotProduct(s[0]);

        switch (method) {
            case DAVIDON_FLETCHER_POWELL:
                currentH = previousH.subtract(previousH.multiply(
                        y[0].mapMultiply(1 / previousH.preMultiply(y[0]).dotProduct(y[0]))
                                .outerProduct(y[0])
                                .multiply(previousH)
                )).add(s[0].mapMultiply(rho).outerProduct(s[0]));
                break;
            case BROYDEN_FLETCHER_GOLDFARB_SHANNO:
                currentH = previousH
                        .preMultiply(identityMatrix.subtract(s[0].mapMultiply(rho).outerProduct(y[0])))
                        .multiply(identityMatrix.subtract(y[0].mapMultiply(rho).outerProduct(s[0])))
                        .add(s[0].mapMultiply(rho).outerProduct(s[0]));
                break;
            case SYMMETRIC_RANK_ONE:
                RealVector tempVector = s[0].subtract(previousH.operate(y[0]));
                if (Math.abs(tempVector.dotProduct(y[0]))
                        >= symmetricRankOneSkippingParameter * y[0].getNorm() * tempVector.getNorm()) {
                    currentH = previousH.add(
                            tempVector.mapMultiply(1 / tempVector.dotProduct(y[0])).outerProduct(tempVector)
                    );
                } else {
                    currentH = previousH;
                }
                break;
            case BROYDEN:
                currentH = previousH.add(
                        s[0].subtract(previousH.operate(y[0]))
                                .outerProduct(previousH.preMultiply(s[0]))
                                .scalarMultiply(1 / previousH.preMultiply(s[0]).dotProduct(y[0]))
                );
                break;
            default:
                throw new NotImplementedException();
        }
    }

    /** Used only for the LBFGS method. */
    private RealVector approximateHessianInverseVectorProduct(RealVector q) {
        double[] a = new double[m];
        double[] rho = new double[m];
        for (int i = 0; i < Math.min(m, currentIteration); i++) {
            rho[i] = 1 / y[i].dotProduct(s[i]);
            a[i] = rho[i] * s[i].dotProduct(q);
            q = q.subtract(y[i].mapMultiply(a[i]));
        }
        RealVector result = q.ebeMultiply(initialHessianInverseDiagonal);
        for (int i = Math.min(m, currentIteration) - 1; i >= 0; i--) {
            result = result.add(s[i].mapMultiply(a[i] - rho[i] * y[i].dotProduct(result)));
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

    public Method getMethod() {
        return method;
    }

    public void setMethod(Method method) {
        this.method = method;
    }

    public int getM() {
        return m;
    }

    public void setM(int m) {
        this.m = m;
    }

    public double getSymmetricRankOneSkippingParameter() {
        return symmetricRankOneSkippingParameter;
    }

    public void setSymmetricRankOneSkippingParameter(double symmetricRankOneSkippingParameter) {
        this.symmetricRankOneSkippingParameter = symmetricRankOneSkippingParameter;
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
