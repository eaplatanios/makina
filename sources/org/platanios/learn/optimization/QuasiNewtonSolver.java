package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.*;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.linesearch.LineSearch;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Emmanouil Antonios Platanios
 */
public class QuasiNewtonSolver extends AbstractLineSearchSolver {
    private final QuasiNewtonMethod method;
    private final RealMatrix identityMatrix;

    private RealMatrix currentH;
    private RealMatrix previousH;

    private double symmetricRankOneSkippingParameter = 1e-8;

    /**
     * BROYDEN_FLETCHER_GOLDFARB_SHANNO method chosen by default.
     *
     * @param objective
     * @param initialPoint
     */
    public QuasiNewtonSolver(Function objective,
                             double[] initialPoint) {
        this(objective,
             initialPoint,
             QuasiNewtonMethod.BROYDEN_FLETCHER_GOLDFARB_SHANNO,
             new StrongWolfeInterpolationLineSearch(objective,
                                                    StepSizeInitializationMethod.UNIT,
                                                    1e-4,
                                                    0.9,
                                                    1000));
    }

    public QuasiNewtonSolver(Function objective,
                             double[] initialPoint,
                             QuasiNewtonMethod method) {
        this(objective,
             initialPoint,
             method,
             new StrongWolfeInterpolationLineSearch(objective,
                                                    StepSizeInitializationMethod.UNIT,
                                                    1e-4,
                                                    0.9,
                                                    1000));
    }

    public QuasiNewtonSolver(Function objective,
                             double[] initialPoint,
                             QuasiNewtonMethod method,
                             LineSearch lineSearch) {
        super(objective, initialPoint);
        this.method = method;
        setLineSearch(lineSearch);
        identityMatrix = MatrixUtils.createRealIdentityMatrix(initialPoint.length);
        currentH = identityMatrix;
        currentGradient = objective.computeGradient(currentPoint);
    }

    public void updateDirection() {
        currentDirection = currentH.operate(currentGradient).mapMultiply(-1);
    }

    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
        currentGradient = objective.computeGradient(currentPoint);
        RealVector s = currentPoint.subtract(previousPoint);
        RealVector y = currentGradient.subtract(previousGradient);
        double rho = 1 / y.dotProduct(s);

        // Simple trick to initialize the inverse Hessian matrix approximation.
        if (currentIteration == 0) {
            previousH = currentH.scalarMultiply(y.dotProduct(s) / y.dotProduct(y));
        } else {
            previousH = currentH;
        }

        switch (method) {
            case DAVIDON_FLETCHER_POWELL:
                currentH = previousH.subtract(
                        previousH.multiply(
                                y.mapMultiply(1 / previousH.preMultiply(y).dotProduct(y))
                                        .outerProduct(y)
                                        .multiply(previousH)
                        )
                ).add(s.mapMultiply(rho).outerProduct(s));
                break;
            case BROYDEN_FLETCHER_GOLDFARB_SHANNO:
                currentH =
                        previousH.preMultiply(
                                identityMatrix.subtract(s.mapMultiply(rho).outerProduct(y))
                        ).multiply(
                                identityMatrix.subtract(y.mapMultiply(rho).outerProduct(s))
                        ).add(s.mapMultiply(rho).outerProduct(s));
                break;
            case SYMMETRIC_RANK_ONE:
                RealVector tempVector = s.subtract(previousH.operate(y));
                if (Math.abs(tempVector.dotProduct(y))
                        >= symmetricRankOneSkippingParameter * y.getNorm() * tempVector.getNorm()) {
                    currentH = previousH.add(
                            tempVector.mapMultiply(1 / tempVector.dotProduct(y)).outerProduct(tempVector)
                    );
                } else {
                    currentH = previousH;
                }
                break;
            default:
                throw new NotImplementedException();
        }
    }

    public double getSymmetricRankOneSkippingParameter() {
        return symmetricRankOneSkippingParameter;
    }

    public void setSymmetricRankOneSkippingParameter(double symmetricRankOneSkippingParameter) {
        this.symmetricRankOneSkippingParameter = symmetricRankOneSkippingParameter;
    }
}
