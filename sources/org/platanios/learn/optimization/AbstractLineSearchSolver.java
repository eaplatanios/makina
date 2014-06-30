package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;

/**
 * @author Emmanouil Antonios Platanios
 */
abstract class AbstractLineSearchSolver extends AbstractSolver {
    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    public AbstractLineSearchSolver(AbstractFunction objective,
                                    double[] initialPoint) {
        super(objective, initialPoint);

        if (objective instanceof QuadraticFunction) {
            RealMatrix quadraticFactorMatrix = ((QuadraticFunction) objective).getA();
            if (MatrixUtils.isSymmetric(quadraticFactorMatrix, 1e-8)) {
                DecompositionSolver choleskyDecompositionSolver =
                        new CholeskyDecomposition(quadraticFactorMatrix).getSolver();
                if (choleskyDecompositionSolver.isNonSingular()) {
                    lineSearch = new ExactLineSearch((QuadraticFunction) objective);
                    return;
                }
            }
        }

        lineSearch = new StrongWolfeInterpolationLineSearch(objective, 1e-4, 0.9, 10);
        ((StrongWolfeInterpolationLineSearch) lineSearch)
                .setStepSizeInitializationMethod(StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE);
    }

    @Override
    public void iterationUpdate() {
        previousGradient = currentGradient;
        previousDirection = currentDirection;
        updateDirection();
        previousStepSize = currentStepSize;
        updateStepSize();
        previousPoint = currentPoint;
        previousObjectiveValue = currentObjectiveValue;
        updatePoint();
        currentObjectiveValue = objective.getValue(currentPoint);
    }

    public void updateStepSize() {
        currentStepSize = lineSearch.computeStepSize(currentPoint,
                                                     currentDirection,
                                                     previousPoint,
                                                     previousDirection,
                                                     previousStepSize);
    }

    public abstract void updateDirection();
    public abstract void updatePoint();

    public LineSearch getLineSearch() {
        return lineSearch;
    }

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }
}
