package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.QuadraticFunction;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ConjugateGradientSolver extends AbstractSolver {
    final QuadraticFunction objective;
    final RealMatrix A;
    final RealVector b;

    RealVector previousPoint;
    RealVector currentResidual;
    RealVector previousResidual;
    RealVector currentDirection;
    RealVector previousDirection;
    double currentObjectiveValue;

    public ConjugateGradientSolver(QuadraticFunction objective,
                                   double[] initialPoint) {
        this.objective = objective;
        A = objective.getA();
        b = objective.getB();
        currentPoint = new ArrayRealVector(initialPoint);
        currentResidual = computeResidual(currentPoint);
        currentDirection = currentResidual.mapMultiply(-1);
        currentObjectiveValue = objective.computeValue(currentPoint);
        currentIteration = 0;
    }

    private RealVector computeResidual(RealVector point) {
        return A.operate(point).subtract(b);
    }

    public boolean checkTerminationConditions() {
        return currentResidual.getNorm() == 0;
    }

    public void iterationUpdate() {
        previousPoint = currentPoint;
        previousResidual = currentResidual;
        previousDirection = currentDirection;
        double previousResidualNormSquared = previousResidual.dotProduct(previousResidual);
        double stepSize = previousResidualNormSquared / A.preMultiply(previousDirection).dotProduct(previousDirection);
        currentPoint = previousPoint.add(currentDirection.mapMultiply(stepSize));
        currentResidual = previousResidual.add(A.operate(previousDirection).mapMultiply(stepSize));
        double residualNormsRatio = currentResidual.dotProduct(currentResidual) / previousResidualNormSquared;
        currentDirection = currentResidual.mapMultiply(-1).add(previousDirection.mapMultiply(residualNormsRatio));
        currentObjectiveValue = objective.computeValue(currentPoint); // TODO: Not necessary to compute the objective function value at each iteration.
    }

    public void printHeader() {
        System.out.println("Iteration #\tObjective Value\tPoint");
        System.out.println("===========\t===============\t=====");
    }

    public void printIteration() {
        System.out.format("%d\t\t\t%.10f\t%.5f\n", currentIteration, currentObjectiveValue, currentPoint.getEntry(0));
    }

    public void printTerminationMessage() {
        System.out.println("The residual became equal to 0 and thus the solution has been found!");
    }
}
