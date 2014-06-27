package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractSolver implements Solver {
    int currentIteration;
    RealVector currentPoint;
    RealVector previousPoint;
    RealVector currentGradient;
    RealVector previousGradient;
    RealVector currentDirection;
    RealVector previousDirection;
    double currentStepSize;
    double previousStepSize;
    double currentObjectiveValue;
    double previousObjectiveValue;

    double pointChangeTolerance = 1e-10;
    double objectiveChangeTolerance = 1e-10;
    double gradientTolerance = 1e-3;

    boolean checkForPointConvergence = true;
    boolean checkForObjectiveConvergence = true;
    boolean checkForGradientConvergence = true;

    double pointChange;
    double objectiveChange;
    double gradientNorm;

    boolean pointConverged = false;
    boolean objectiveConverged = false;
    boolean gradientConverged = false;

    public RealVector solve() {
        printHeader();
        while (!checkTerminationConditions()) {
            iterationUpdate();
            currentIteration++;
            printIteration();
        }

        printTerminationMessage();

        return currentPoint;
    }

    public boolean checkTerminationConditions() {
        if (currentIteration > 0) {
            pointChange = currentPoint.subtract(previousPoint).getNorm();
            pointConverged = pointChange <= pointChangeTolerance;
            objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
            objectiveConverged = objectiveChange <= objectiveChangeTolerance;

            if (this instanceof NonlinearConjugateGradientSolver) {
                gradientNorm = currentGradient.getNorm();
                gradientConverged = gradientNorm <= gradientTolerance;
            } else {
                gradientNorm = currentDirection.getNorm();
                gradientConverged = gradientNorm <= gradientTolerance;
            }

            return (checkForPointConvergence && pointConverged)
                    || (checkForObjectiveConvergence && objectiveConverged)
                    || (checkForGradientConvergence && gradientConverged);
        } else {
            return false;
        }
    }

    public void printTerminationMessage() {
        if (pointConverged) {
            System.out.println("The L2 norm of the point change, "
                                       + DECIMAL_FORMAT.format(pointChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(pointChangeTolerance)
                                       + "!\n");
        }
        if (objectiveConverged) {
            System.out.println("The relative change of the objective function value, "
                                       + DECIMAL_FORMAT.format(objectiveChange)
                                       + ", was below the convergence threshold of "
                                       + DECIMAL_FORMAT.format(objectiveChangeTolerance)
                                       + "!\n");
        }
        if (gradientConverged) {
            System.out.println("The gradient norm became "
                                       + DECIMAL_FORMAT.format(gradientNorm)
                                       + ", which is less than the convergence threshold of "
                                       + DECIMAL_FORMAT.format(gradientTolerance)
                                       + "!\n");
        }
    }

    public double getPointChangeTolerance() {
        return pointChangeTolerance;
    }

    public void setPointChangeTolerance(double pointChangeTolerance) {
        this.pointChangeTolerance = pointChangeTolerance;
    }

    public double getObjectiveChangeTolerance() {
        return objectiveChangeTolerance;
    }

    public void setObjectiveChangeTolerance(double objectiveChangeTolerance) {
        this.objectiveChangeTolerance = objectiveChangeTolerance;
    }

    public double getGradientTolerance() {
        return gradientTolerance;
    }

    public void setGradientTolerance(double gradientTolerance) {
        this.gradientTolerance = gradientTolerance;
    }

    public boolean isCheckForPointConvergence() {
        return checkForPointConvergence;
    }

    public void setCheckForPointConvergence(boolean checkForPointConvergence) {
        this.checkForPointConvergence = checkForPointConvergence;
    }

    public boolean isCheckForObjectiveConvergence() {
        return checkForObjectiveConvergence;
    }

    public void setCheckForObjectiveConvergence(boolean checkForObjectiveConvergence) {
        this.checkForObjectiveConvergence = checkForObjectiveConvergence;
    }

    public boolean isCheckForGradientConvergence() {
        return checkForGradientConvergence;
    }

    public void setCheckForGradientConvergence(boolean checkForGradientConvergence) {
        this.checkForGradientConvergence = checkForGradientConvergence;
    }
}
