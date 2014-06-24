package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractSolver implements Solver {
    final Function objectiveFunction;
    final LineSearch lineSearch;
    final DecimalFormat decimalFormat;

    int currentIteration;
    RealVector currentPoint;
    RealVector previousPoint;
    RealVector currentDirection;
    double currentObjectiveValue;
    double previousObjectiveValue;
    List<Double> stepSizes;
    double pointL2NormChange;
    double objectiveChange;
    double pointL2NormChangeTolerance = 1e-10;
    double objectiveChangeTolerance = 1e-10;
    boolean pointL2NormConverged = false;
    boolean gradientConverged = false;
    boolean objectiveConverged = false;

    public AbstractSolver(Function objectiveFunction,
                          double[] initialPoint) {
        this.objectiveFunction = objectiveFunction;
        this.currentPoint = new ArrayRealVector(initialPoint);
        currentObjectiveValue = objectiveFunction.computeValue(currentPoint);
        decimalFormat = new DecimalFormat("0.##E0");
        currentIteration = 0;
        stepSizes = new ArrayList<>();

        if (objectiveFunction.getClass() == QuadraticFunction.class) {
            this.lineSearch = new ExactLineSearch((QuadraticFunction) objectiveFunction);
        } else {
            this.lineSearch =
                    new BacktrackingLineSearch(objectiveFunction, 1.0, 0.9, 1e-4);
        }
    }

    public AbstractSolver(Function objectiveFunction,
                          double[] initialPoint,
                          LineSearch lineSearch) {
        this.objectiveFunction = objectiveFunction;
        this.currentPoint = new ArrayRealVector(initialPoint);
        this.lineSearch = lineSearch;
        currentObjectiveValue = objectiveFunction.computeValue(currentPoint);
        decimalFormat = new DecimalFormat("0.##E0");
        currentIteration = 0;
        stepSizes = new ArrayList<>();
    }

    public void updateStepSize() {
        stepSizes.add(lineSearch.computeStepSize(currentPoint, currentDirection));
    }

    public boolean checkForConvergence() {
        pointL2NormChange = currentPoint.subtract(previousPoint).getNorm();
        objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
        pointL2NormConverged = pointL2NormChange <= pointL2NormChangeTolerance;
        objectiveConverged = objectiveChange <= objectiveChangeTolerance;
        gradientConverged = objectiveFunction.computeGradient(currentPoint).getNorm() == 0.0;

        return pointL2NormConverged || objectiveConverged || gradientConverged;
    }

    public RealVector solve() {
        do {
            previousPoint = currentPoint;
            previousObjectiveValue = currentObjectiveValue;
            updateDirection();
            updateStepSize();
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
        if (gradientConverged) {
            System.out.println("The gradient became 0!\n");
        }

        return currentPoint;
    }
}
