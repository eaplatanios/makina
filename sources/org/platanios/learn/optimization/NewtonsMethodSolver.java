package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NewtonsMethodSolver extends AbstractSolver {
    public NewtonsMethodSolver(ObjectiveFunctionWithGradientAndHessian objectiveFunction,
                                 double[] initialPoint) {
        super(objectiveFunction, new BacktrackingLineSearchAlgorithm(objectiveFunction, 1.0, 0.9, 1e-4), initialPoint);
    }

    public void updateDirection() {
        RealMatrix hessian = ((ObjectiveFunctionWithGradientAndHessian) objectiveFunction).computeHessian(currentPoint);
        RealVector gradient =
                ((ObjectiveFunctionWithGradientAndHessian) objectiveFunction).computeGradient(currentPoint);
        currentDirection = new LUDecomposition(hessian).getSolver().getInverse().operate(gradient).mapMultiply(-1);
    }

    public void updatePoint() {
        double stepSize = lineSearchAlgorithm.computeStepSize(currentPoint, currentDirection);
        currentPoint = currentPoint.add(currentDirection.mapMultiply(stepSize));
    }

    public boolean checkForConvergence() {
        pointL2NormChange = currentPoint.subtract(previousPoint).getNorm();
        objectiveChange = Math.abs((previousObjectiveValue - currentObjectiveValue) / previousObjectiveValue);
        pointL2NormConverged = pointL2NormChange <= pointL2NormChangeTolerance;
        objectiveConverged = objectiveChange <= objectiveChangeTolerance;

        return pointL2NormConverged || objectiveConverged;
    }
}
