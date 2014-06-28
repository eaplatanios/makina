package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
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
public class ConjugateGradientSolver extends AbstractSolver {
    private final QuadraticFunction objective;
    private final RealMatrix A;

    private RealVector previousPoint;
    private RealVector currentResidual;
    private RealVector previousResidual;
    private RealVector currentDirection;
    private RealVector previousDirection;

    private double residualTolerance = 1e-10;

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   double[] initialPoint) {
        this.objective = objective;
        A = objective.getA();
        currentPoint = new ArrayRealVector(initialPoint);
        currentResidual = objective.computeGradient(currentPoint);
        currentDirection = currentResidual.mapMultiply(-1);
        currentObjectiveValue = objective.computeValue(currentPoint);
        currentIteration = 0;
    }

    public boolean checkTerminationConditions() {
        return currentResidual.getNorm() <= residualTolerance;
    }

    public void iterationUpdate() {
        previousPoint = currentPoint;
        previousResidual = currentResidual;
        previousDirection = currentDirection;
        double previousResidualNormSquared = previousResidual.dotProduct(previousResidual);
        double stepSize = previousResidualNormSquared / A.preMultiply(previousDirection).dotProduct(previousDirection);
        currentPoint = previousPoint.add(previousDirection.mapMultiply(stepSize));
        currentResidual = previousResidual.add(A.operate(previousDirection).mapMultiply(stepSize));
        double residualNormsRatio = currentResidual.dotProduct(currentResidual) / previousResidualNormSquared;
        currentDirection = currentResidual.mapMultiply(-1).add(previousDirection.mapMultiply(residualNormsRatio));
        currentObjectiveValue = objective.computeValue(currentPoint);
    }

    public double getResidualTolerance() {
        return residualTolerance;
    }

    public void setResidualTolerance(double residualTolerance) {
        this.residualTolerance = residualTolerance;
    }
}
