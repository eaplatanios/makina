package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.RealMatrix;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * Matrix A in this case needs to be symmetric and positive definite. The biconjugate gradient does not have that
 * requirement.
 *
 * TODO: Implement the preconditioned conjugate gradient method and a few preconditioning strategies. Support for
 * preconditioning should be added as a flag to this algorithm, possibly with an extra argument in the constructor for
 * the preconditioning matrix or the preconditioning method.
 *
 * @author Emmanouil Antonios Platanios
 */
public class ConjugateGradientSolver extends AbstractIterativeSolver {
    private final RealMatrix A;

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   double[] initialPoint) {
        super(objective, initialPoint);
        A = objective.getA();
        currentDirection = currentGradient.mapMultiply(-1);
    }

    @Override
    public void iterationUpdate() {
        previousPoint = currentPoint;
        previousGradient = currentGradient;
        previousDirection = currentDirection;
        double previousResidualNormSquared = previousGradient.dotProduct(previousGradient);
        double stepSize = previousResidualNormSquared / A.preMultiply(previousDirection).dotProduct(previousDirection);
        currentPoint = previousPoint.add(previousDirection.mapMultiply(stepSize));
        currentGradient = previousGradient.add(A.operate(previousDirection).mapMultiply(stepSize));
        double residualNormsRatio = currentGradient.dotProduct(currentGradient) / previousResidualNormSquared;
        currentDirection = currentGradient.mapMultiply(-1).add(previousDirection.mapMultiply(residualNormsRatio));
        currentObjectiveValue = objective.getValue(currentPoint);
    }
}
