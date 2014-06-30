package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NewtonSolver extends AbstractLineSearchSolver {
    public NewtonSolver(AbstractFunction objective,
                        double[] initialPoint) {
        super(objective, initialPoint);
        setLineSearch(new StrongWolfeInterpolationLineSearch(objective,
                                                             StepSizeInitializationMethod.UNIT,
                                                             1e-4,
                                                             0.9,
                                                             1));
    }

    /**
     * Here, if the Hessian matrix is not positive definite, we modify it so that the bounded modified factorization
     * property holds for it and we have global convergence for Newton's method.
     */
    @Override
    public void updateDirection() {
        RealMatrix hessian = objective.getHessian(currentPoint);
        // TODO: Check Hessian for positive definiteness and modify if necessary.
        currentGradient = objective.getGradient(currentPoint);
        currentDirection =
                new LUDecomposition(hessian).getSolver().getInverse().operate(currentGradient).mapMultiply(-1);
    }

    @Override
    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
    }
}
