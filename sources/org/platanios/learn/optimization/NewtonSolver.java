package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.linesearch.LineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NewtonSolver extends AbstractLineSearchSolver {
    public NewtonSolver(Function objective,
                        double[] initialPoint) {
        super(objective, initialPoint);
    }

    public NewtonSolver(Function objective,
                        double[] initialPoint,
                        LineSearch lineSearch) {
        super(objective, initialPoint);
        setLineSearch(lineSearch);
    }

    /**
     * Here, if the Hessian matrix is not positive definite, we modify it so that the bounded modified factorization
     * property holds for it and we have global convergence for Newton's method.
     */
    public void updateDirection() {
        RealMatrix hessian = objective.computeHessian(currentPoint);
        // TODO: Check Hessian for positive definiteness and modify if necessary.
        currentGradient = objective.computeGradient(currentPoint);
        currentDirection =
                new LUDecomposition(hessian).getSolver().getInverse().operate(currentGradient).mapMultiply(-1);
    }

    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
    }
}
