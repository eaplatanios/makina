package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;

import java.text.DecimalFormat;

/**
 * @author Emmanouil Antonios Platanios
 */
public abstract class AbstractSolver implements Solver {
    final Function objectiveFunction;
    final DecimalFormat decimalFormat;

    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is BacktrackingLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    int currentIteration;
    RealVector currentDirection;
    RealVector previousDirection;
    double currentStepSize;
    double previousStepSize;
    StepSizeInitializationMethod stepSizeInitializationMethod;
    RealVector currentPoint;
    RealVector previousPoint;
    double currentObjectiveValue;
    double previousObjectiveValue;
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

        if (objectiveFunction.getClass() == QuadraticFunction.class) {
            this.lineSearch = new ExactLineSearch((QuadraticFunction) objectiveFunction);
        } else {
            stepSizeInitializationMethod = StepSizeInitializationMethod.UNIT;
//            this.lineSearch =
//                    new BacktrackingLineSearch(objectiveFunction,
//                                               stepSizeInitializationMethod,
//                                               0.9,
//                                               1e-4);
            this.lineSearch = new ArmijoInterpolationLineSearch(objectiveFunction, stepSizeInitializationMethod, 1e-4);
        }
    }

    public void updateStepSize() {
        if (!(lineSearch instanceof IterativeLineSearch)) {
            currentStepSize = lineSearch.computeStepSize(currentPoint, currentDirection);
            return;
        }

        switch (stepSizeInitializationMethod) {
            case UNIT: case CONSTANT:
                currentStepSize = lineSearch.computeStepSize(currentPoint, currentDirection);
                break;
            case CONSERVE_FIRST_ORDER_CHANGE:
                currentStepSize = ((IterativeLineSearch) lineSearch).computeStepSize(
                        currentPoint,
                        currentDirection,
                        previousPoint,
                        previousDirection,
                        previousStepSize
                );
                break;
            case QUADRATIC_INTERPOLATION:
                currentStepSize = ((IterativeLineSearch) lineSearch).computeStepSize(
                        currentPoint,
                        currentDirection,
                        previousPoint,
                        previousDirection
                );
                break;
        }
    }

    public boolean checkForConvergence() {
        pointL2NormChange = currentPoint.subtract(previousPoint).getNorm();
        objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
        pointL2NormConverged = pointL2NormChange <= pointL2NormChangeTolerance;
        objectiveConverged = objectiveChange <= objectiveChangeTolerance;
        gradientConverged = objectiveFunction.computeGradient(currentPoint).getNorm() == 0.0;

        return pointL2NormConverged || objectiveConverged || gradientConverged;
    }

    public void printHeader() {
        System.out.println("Iteration #\tObjective Value\tPoint");
        System.out.println("===========\t===============\t=====");
    }

    public void printIteration() {
        System.out.format("%d\t\t\t%.10f\t%.5f\n", currentIteration, currentObjectiveValue, currentPoint.getEntry(0));
    }

    public RealVector solve() {
        printHeader();
        do {
            updateDirection();
            updateStepSize();

            previousDirection = currentDirection;
            previousStepSize = currentStepSize;
            previousPoint = currentPoint;
            previousObjectiveValue = currentObjectiveValue;

            updatePoint();

            currentObjectiveValue = objectiveFunction.computeValue(currentPoint);
            currentIteration++;

            printIteration();
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

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }
}
