package org.platanios.learn.optimization;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.platanios.learn.optimization.function.Function;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.*;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * @author Emmanouil Antonios Platanios
 */
class NonlinearConjugateGradientSolver extends AbstractSolver {
    private final Function objective;
    private final NonlinearConjugateGradientMethod method;

    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private LineSearch lineSearch;

    // The following variables are used locally within iteration but are initialized here in order to make the code more
    // clear.
    double beta;

    public NonlinearConjugateGradientSolver(Function objective,
                                            double[] initialPoint) {
        this(objective, initialPoint, NonlinearConjugateGradientMethod.POLAK_RIBIERE_PLUS);
    }

    public NonlinearConjugateGradientSolver(Function objective,
                                            double[] initialPoint,
                                            NonlinearConjugateGradientMethod method) {
        this.objective = objective;
        this.method = method;
        currentPoint = new ArrayRealVector(initialPoint);
        currentGradient = objective.computeGradient(currentPoint);
        currentDirection = currentGradient.mapMultiply(-1);
        currentObjectiveValue = objective.computeValue(currentPoint);
        currentIteration = 0;
        checkForPointConvergence = false;
        checkForObjectiveConvergence = false;

        if (objective instanceof QuadraticFunction) {
            this.lineSearch = new ExactLineSearch((QuadraticFunction) objective);
        } else {
            this.lineSearch = new StrongWolfeInterpolationLineSearch(
                    objective,
                    StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE,
                    1e-4,
                    0.1,
                    10
            );
        }
    }

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
        currentGradient = objective.computeGradient(currentPoint);
        beta = computeBeta();
        currentDirection = currentGradient.mapMultiply(-1).add(previousDirection.mapMultiply(beta));
        currentObjectiveValue = objective.computeValue(currentPoint);
    }

    public double computeBeta() {
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

    public void printHeader() {
        System.out.println("Iteration #\tObjective Value\tPoint");
        System.out.println("===========\t===============\t=====");
    }

    public void printIteration() {
        System.out.format("%d\t\t\t%.10f\t%.5f\n", currentIteration, currentObjectiveValue, currentPoint.getEntry(0));
    }

    public LineSearch getLineSearch() {
        return lineSearch;
    }

    public void setLineSearch(LineSearch lineSearch) {
        this.lineSearch = lineSearch;
    }
}
