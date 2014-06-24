package org.platanios.learn.optimization;

import org.platanios.learn.math.linearalgebra.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SteepestDescentSolver extends AbstractSolver {
    public SteepestDescentSolver(ObjectiveFunctionWithGradient objectiveFunction,
                                 double[] initialPoint) {
        super(objectiveFunction, new BacktrackingLineSearchAlgorithm(objectiveFunction, 1.0, 0.9, 1e-4), initialPoint);
    }

    public void updateDirection() {
        currentDirection = Vector.multiply(
                -1,                ((ObjectiveFunctionWithGradient) objectiveFunction).computeGradient(currentPoint)
        );
    }

    public void updatePoint() {
        double stepSize = lineSearchAlgorithm.computeStepSize(currentPoint, currentDirection);
        currentPoint = Vector.add(currentPoint, Vector.multiply(stepSize, currentDirection));
    }

    public boolean checkForConvergence() {
        pointL2NormChange = Vector.l2Norm(Vector.subtract(currentPoint, previousPoint));
        objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
        pointL2NormConverged = pointL2NormChange <= pointL2NormChangeTolerance;
        objectiveConverged = objectiveChange <= objectiveChangeTolerance;

        return pointL2NormConverged || objectiveConverged;
    }

    public double[] solve() {
        do {
            previousPoint = currentPoint;
            previousObjectiveValue = currentObjectiveValue;
            updateDirection();
            updatePoint();
            currentObjectiveValue = objectiveFunction.computeValue(currentPoint);

            System.out.format("Current objective value: %.10f\n", currentObjectiveValue);
        } while (!checkForConvergence());

        if (pointL2NormConverged) {
            System.out.println("The L2 norm of the point change, "
                    + decimalFormat.format(pointL2NormChange)
                    + ", was below the convergence threshold of "
                    + decimalFormat.format(pointL2NormChangeTolerance)
                    + "!\n");
        }
        if (objectiveConverged) {
            System.out.println("The relative change of the objective function value, "
                    + decimalFormat.format(objectiveChange)
                    + ", was below the convergence threshold of "
                    + decimalFormat.format(objectiveChangeTolerance)
                    + "!\n");
        }

        return currentPoint;
    }
}
