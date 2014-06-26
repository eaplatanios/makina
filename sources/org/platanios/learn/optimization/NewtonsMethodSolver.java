package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.linesearch.LineSearch;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeLineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NewtonsMethodSolver extends AbstractLineSearchSolver {
    public NewtonsMethodSolver(Function objective,
                               double[] initialPoint) {
        super(objective, initialPoint);
        setLineSearch(
                new StrongWolfeLineSearch(
                        objective,
                        StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE,
                        1e-4,
                        0.9,
                        10
                )
        );
    }

    public NewtonsMethodSolver(Function objective,
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
        RealVector gradient = objective.computeGradient(currentPoint);
        currentDirection = new LUDecomposition(hessian).getSolver().getInverse().operate(gradient).mapMultiply(-1);
    }

    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
    }
}
