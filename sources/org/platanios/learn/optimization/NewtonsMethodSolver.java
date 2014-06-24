package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NewtonsMethodSolver extends AbstractSolver {
    public NewtonsMethodSolver(Function objectiveFunction,
                               double[] initialPoint) {
        super(objectiveFunction, initialPoint);
    }

    public NewtonsMethodSolver(Function objectiveFunction,
                               double[] initialPoint,
                               LineSearch lineSearch) {
        super(objectiveFunction, initialPoint, lineSearch);
    }

    /**
     * Here, if the Hessian matrix is not positive definite, we modify it so that the bounded modified factorization
     * property holds for it and we have global convergence for Newton's method.
     */
    public void updateDirection() {
        RealMatrix hessian = objectiveFunction.computeHessian(currentPoint);
        // TODO: Check Hessian for positive definiteness and modify if necessary.
        RealVector gradient = objectiveFunction.computeGradient(currentPoint);
        currentDirection = new LUDecomposition(hessian).getSolver().getInverse().operate(gradient).mapMultiply(-1);
    }

    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(stepSizes.get(currentIteration)));
    }
}
