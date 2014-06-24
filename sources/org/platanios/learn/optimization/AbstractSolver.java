package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.text.DecimalFormat;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractSolver implements Solver {
    final ObjectiveFunction objectiveFunction;
    final LineSearchAlgorithm lineSearchAlgorithm;
    final DecimalFormat decimalFormat;

    int currentIteration;
    RealVector currentPoint;
    RealVector previousPoint;
    RealVector currentDirection;
    double currentObjectiveValue;
    double previousObjectiveValue;
    double pointL2NormChange;
    double objectiveChange;
    double pointL2NormChangeTolerance = 1e-10;
    double objectiveChangeTolerance = 1e-10;
    boolean pointL2NormConverged;
    boolean objectiveConverged;

    public AbstractSolver(ObjectiveFunction objectiveFunction,
                          LineSearchAlgorithm lineSearchAlgorithm,
                          double[] initialPoint) {
        this.objectiveFunction = objectiveFunction;
        this.lineSearchAlgorithm = lineSearchAlgorithm;
        this.currentPoint = new ArrayRealVector(initialPoint);
        currentObjectiveValue = objectiveFunction.computeValue(currentPoint);
        decimalFormat = new DecimalFormat("0.##E0");
        currentIteration = 0;
    }

    public boolean checkForConvergence() {
        pointL2NormChange = currentPoint.subtract(previousPoint).getNorm();
        objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
        pointL2NormConverged = pointL2NormChange <= pointL2NormChangeTolerance;
        objectiveConverged = objectiveChange <= objectiveChangeTolerance;

        return pointL2NormConverged || objectiveConverged;
    }

    public RealVector solve() {
        do {
            previousPoint = currentPoint;
            previousObjectiveValue = currentObjectiveValue;
            updateDirection();
            updatePoint();
            currentObjectiveValue = objectiveFunction.computeValue(currentPoint);
            currentIteration++;

            System.out.format("Iteration #%d: Current objective value: %.10f\n", currentIteration, currentObjectiveValue);
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
