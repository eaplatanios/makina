package org.platanios.learn.optimization;

import org.platanios.learn.optimization.function.AbstractLeastSquaresFunction;
import org.platanios.learn.optimization.function.LinearLeastSquaresFunction;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GaussNewtonSolver extends AbstractLineSearchSolver {
    private LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod =
            LinearLeastSquaresSolver.Method.SINGULAR_VALUE_DECOMPOSITION;

    public GaussNewtonSolver(AbstractLeastSquaresFunction objective,
                             double[] initialPoint) {
        super(objective, initialPoint);
        StrongWolfeInterpolationLineSearch lineSearch = new StrongWolfeInterpolationLineSearch(objective, 1e-4, 0.9, 1);
        lineSearch.setStepSizeInitializationMethod(StepSizeInitializationMethod.UNIT);
        setLineSearch(lineSearch);
    }

    @Override
    public void updateDirection() {
        LinearLeastSquaresSolver linearLeastSquaresSubproblemSolver =
                new LinearLeastSquaresSolver(new LinearLeastSquaresFunction(
                        ((AbstractLeastSquaresFunction) objective).computeJacobian(currentPoint),
                        ((AbstractLeastSquaresFunction) objective).computeResiduals(currentPoint).mapMultiply(-1)
                ));
        linearLeastSquaresSubproblemSolver.setMethod(linearLeastSquaresSubproblemMethod);
        currentDirection = linearLeastSquaresSubproblemSolver.solve();
    }

    @Override
    public void updatePoint() {
        currentPoint = currentPoint.add(currentDirection.mapMultiply(currentStepSize));
    }

    public LinearLeastSquaresSolver.Method getLinearLeastSquaresSubproblemMethod() {
        return linearLeastSquaresSubproblemMethod;
    }

    public void setLinearLeastSquaresSubproblemMethod(LinearLeastSquaresSolver.Method linearLeastSquaresSubproblemMethod) {
        this.linearLeastSquaresSubproblemMethod = linearLeastSquaresSubproblemMethod;
    }
}
