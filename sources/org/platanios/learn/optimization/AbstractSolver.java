package org.platanios.learn.optimization;

import org.platanios.learn.math.linearalgebra.Vector;

import java.text.DecimalFormat;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractSolver implements Solver {
    final ObjectiveFunction objectiveFunction;
    final LineSearchAlgorithm lineSearchAlgorithm;
    final DecimalFormat decimalFormat;

    double[] currentPoint;
    double[] previousPoint;
    double[] currentDirection;
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
        this.currentPoint = initialPoint;
        currentObjectiveValue = objectiveFunction.computeValue(currentPoint);
        decimalFormat = new DecimalFormat("0.##E0");
    }

    public boolean checkForConvergence() {
        pointL2NormChange = Vector.l2Norm(Vector.subtract(currentPoint, previousPoint));
        objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
        pointL2NormConverged = pointL2NormChange <= pointL2NormChangeTolerance;
        objectiveConverged = objectiveChange <= objectiveChangeTolerance;

        return pointL2NormConverged || objectiveConverged;
    }
}
