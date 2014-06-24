package org.platanios.learn.optimization;

import org.platanios.learn.math.linearalgebra.Vector;

import java.text.DecimalFormat;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SteepestDescentAlgorithm {
    private final ObjectiveFunction objectiveFunction;
    private final LineSearchAlgorithm lineSearchAlgorithm;
    private final DecimalFormat decimalFormat;

    private double[] currentPoint;
    private double[] previousPoint;
    private double[] currentDirection;
    private double currentObjectiveValue;
    private double previousObjectiveValue;
    private double pointL2NormChange;
    private double objectiveChange;
    private double pointL2NormChangeTolerance = 1e-10;
    private double objectiveChangeTolerance = 1e-10;
    private boolean pointL2NormConverged;
    private boolean objectiveConverged;

    public SteepestDescentAlgorithm(ObjectiveFunction objectiveFunction,
                                    double[] initialPoint) {
        this.objectiveFunction = objectiveFunction;
        this.currentPoint = initialPoint;
        currentObjectiveValue = objectiveFunction.computeValue(currentPoint);

        lineSearchAlgorithm = new BacktrackingLineSearchAlgorithm(objectiveFunction, 1.0, 0.9, 1e-4);
        decimalFormat = new DecimalFormat("0.##E0");
    }

    public void updateDirection() {
        currentDirection = Vector.multiply(-1, objectiveFunction.computeGradient(currentPoint));
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

    public double getPointL2NormChangeTolerance() {
        return pointL2NormChangeTolerance;
    }

    public void setPointL2NormChangeTolerance(double pointL2NormChangeTolerance) {
        this.pointL2NormChangeTolerance = pointL2NormChangeTolerance;
    }

    public double getObjectiveChangeTolerance() {
        return objectiveChangeTolerance;
    }

    public void setObjectiveChangeTolerance(double objectiveChangeTolerance) {
        this.objectiveChangeTolerance = objectiveChangeTolerance;
    }
}
