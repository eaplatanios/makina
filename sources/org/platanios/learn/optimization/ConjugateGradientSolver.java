package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * Matrix A in this case needs to be symmetric and positive definite. The biconjugate gradient does not have that
 * requirement.
 *
 * TODO: Implement the preconditioned conjugate gradient method and a few preconditioning strategies.
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
    private double currentObjectiveValue;

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
    }

    public void printHeader() {
        System.out.println("Iteration #\tObjective Value\tPoint");
        System.out.println("===========\t===============\t=====");
    }

    public void printIteration() {
        currentObjectiveValue = objective.computeValue(currentPoint); // TODO: Not necessary to compute the objective function value at each iteration.
        System.out.format("%d\t\t\t%.10f\t%.5f\n", currentIteration, currentObjectiveValue, currentPoint.getEntry(0));
    }

    public void printTerminationMessage() {
        System.out.println("The residual became equal to 0 and thus the solution has been found!");
    }

    public double getResidualTolerance() {
        return residualTolerance;
    }

    public void setResidualTolerance(double residualTolerance) {
        this.residualTolerance = residualTolerance;
    }
}
