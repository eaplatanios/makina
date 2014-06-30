package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * These solvers are good for large scale problems and, on certain applications, competitive with limited-memory
 * quasi-Newton methods as well.
 *
 * @author Emmanouil Antonios Platanios
 */
public class NonlinearConjugateGradientSolver extends AbstractSolver {
    private NonlinearConjugateGradientMethod method = NonlinearConjugateGradientMethod.POLAK_RIBIERE_PLUS;
    private NonlinearConjugateGradientRestartMethod restartMethod = NonlinearConjugateGradientRestartMethod.GRADIENTS_ORTHOGONALITY_CHECK;

    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    private double gradientsOrthogonalityCheckThreshold = 0.1;

    // The following variables are used locally within iteration but are initialized here in order to make the code more
    // clear.
    double beta;

    public NonlinearConjugateGradientSolver(AbstractFunction objective,
                                            double[] initialPoint) {
        super(objective, initialPoint);
        setCheckForPointConvergence(false);
        setCheckForObjectiveConvergence(false);
        currentDirection = currentGradient.mapMultiply(-1);

        if (objective instanceof QuadraticFunction) {
            lineSearch = new ExactLineSearch((QuadraticFunction) objective);
        } else {
            lineSearch = new StrongWolfeInterpolationLineSearch(
                    objective,
                    StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE,
                    1e-4,
                    0.1,
                    10
            );
        }
    }

    @Override
    public void iterationUpdate() {
        previousPoint = currentPoint;
        previousGradient = currentGradient;
        previousDirection = currentDirection;
        previousStepSize = currentStepSize;
        previousObjectiveValue = currentObjectiveValue;
        currentStepSize = lineSearch.computeStepSize(currentPoint,
                                                     currentDirection,
                                                     previousPoint,
                                                     previousDirection,
                                                     previousStepSize);
        currentPoint = previousPoint.add(previousDirection.mapMultiply(currentStepSize));
        currentGradient = objective.getGradient(currentPoint);
        beta = checkForRestart() ? 0 : computeBeta();
        currentDirection = currentGradient.mapMultiply(-1).add(previousDirection.mapMultiply(beta));
        currentObjectiveValue = objective.getValue(currentPoint);
    }

    private boolean checkForRestart() {
        switch (restartMethod) {
            case NO_RESTART:
                return false;
            case N_STEP:
                return currentIteration % currentPoint.getDimension() == 0;
            case GRADIENTS_ORTHOGONALITY_CHECK:
                return Math.abs(currentGradient.dotProduct(previousGradient))
                        / currentGradient.dotProduct(currentGradient) >= gradientsOrthogonalityCheckThreshold;
            default:
                throw new NotImplementedException();
        }
    }

    private double computeBeta() {
        RealVector gradientsDifference;
        double denominator;

        switch (method) {
            case FLETCHER_RIEVES:
                return currentGradient.dotProduct(currentGradient) / previousGradient.dotProduct(previousGradient);
            case POLAK_RIBIERE:
                return currentGradient.dotProduct(currentGradient.subtract(previousGradient))
                        / previousGradient.dotProduct(previousGradient);
            case POLAK_RIBIERE_PLUS:
                return Math.max(currentGradient.dotProduct(currentGradient.subtract(previousGradient))
                                        / previousGradient.dotProduct(previousGradient), 0);
            case HESTENES_STIEFEL:
                gradientsDifference = currentGradient.subtract(previousGradient);
                return currentGradient.dotProduct(gradientsDifference)
                        / gradientsDifference.dotProduct(previousDirection);
            case FLETCHER_RIEVES_POLAK_RIBIERE:
                denominator = previousGradient.dotProduct(previousGradient);
                double betaFR = currentGradient.dotProduct(currentGradient) / denominator;
                double betaPR = currentGradient.dotProduct(currentGradient.subtract(previousGradient)) / denominator;
                if (betaPR < -betaFR) {
                    return -betaFR;
                } else if (betaPR > betaFR) {
                    return betaFR;
                } else {
                    return betaPR;
                }
            case DAI_YUAN:
                return currentGradient.dotProduct(currentGradient)
                        / currentGradient.subtract(previousGradient).dotProduct(previousDirection);
            case HAGER_ZHANG:
                gradientsDifference = currentGradient.subtract(previousGradient);
                denominator = gradientsDifference.dotProduct(previousDirection);
                RealVector temporaryTerm = gradientsDifference.subtract(
                        previousDirection.mapMultiply(2 * gradientsDifference.dotProduct(gradientsDifference)
                                                              / denominator)
                );
                return temporaryTerm.dotProduct(currentGradient) / denominator;
            default:
                throw new NotImplementedException();
        }
    }

    public NonlinearConjugateGradientMethod getMethod() {
        return method;
    }

    public void setMethod(NonlinearConjugateGradientMethod method) {
        this.method = method;
    }

    public NonlinearConjugateGradientRestartMethod getRestartMethod() {
        return restartMethod;
    }

    public void setRestartMethod(NonlinearConjugateGradientRestartMethod restartMethod) {
        this.restartMethod = restartMethod;
    }

    public LineSearch getLineSearch() {
        return lineSearch;
    }

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }

    public double getGradientsOrthogonalityCheckThreshold() {
        return gradientsOrthogonalityCheckThreshold;
    }

    public void setGradientsOrthogonalityCheckThreshold(double gradientsOrthogonalityCheckThreshold) {
        this.gradientsOrthogonalityCheckThreshold = gradientsOrthogonalityCheckThreshold;
    }
}
