package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.CholeskyDecomposition;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.QuadraticFunction;
import org.platanios.learn.optimization.linesearch.ExactLineSearch;
import org.platanios.learn.optimization.linesearch.LineSearch;
import org.platanios.learn.optimization.linesearch.StepSizeInitializationMethod;
import org.platanios.learn.optimization.linesearch.StrongWolfeInterpolationLineSearch;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * These solvers are good for large scale problems and, on certain applications, competitive with limited-memory
 * quasi-Newton methods as well.
 *
 * @author Emmanouil Antonios Platanios
 */
public class NonlinearConjugateGradientSolver extends AbstractIterativeSolver {
    private Method method = Method.POLAK_RIBIERE_PLUS;
    private RestartMethod restartMethod = RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK;

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
        currentDirection = currentGradient.multiply(-1);

        if (objective instanceof QuadraticFunction) {
            Matrix quadraticFactorMatrix = ((QuadraticFunction) objective).getA();
            CholeskyDecomposition choleskyDecomposition = new CholeskyDecomposition(quadraticFactorMatrix);
            if (choleskyDecomposition.isSymmetricAndPositiveDefinite()) {
                lineSearch = new ExactLineSearch((QuadraticFunction) objective);
                return;
            }
        }

        lineSearch = new StrongWolfeInterpolationLineSearch(objective, 1e-4, 0.9, 10);
        ((StrongWolfeInterpolationLineSearch) lineSearch)
                .setStepSizeInitializationMethod(StepSizeInitializationMethod.CONSERVE_FIRST_ORDER_CHANGE);
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
        currentPoint = previousPoint.add(previousDirection.multiply(currentStepSize));
        currentGradient = objective.getGradient(currentPoint);
        beta = checkForRestart() ? 0 : computeBeta();
        currentDirection = currentGradient.multiply(-1).add(previousDirection.multiply(beta));
        currentObjectiveValue = objective.getValue(currentPoint);
    }

    private boolean checkForRestart() {
        switch (restartMethod) {
            case NO_RESTART:
                return false;
            case N_STEP:
                return currentIteration % currentPoint.getDimension() == 0;
            case GRADIENTS_ORTHOGONALITY_CHECK:
                return Math.abs(currentGradient.innerProduct(previousGradient))
                        / currentGradient.innerProduct(currentGradient) >= gradientsOrthogonalityCheckThreshold;
            default:
                throw new NotImplementedException();
        }
    }

    private double computeBeta() {
        Vector gradientsDifference;
        double denominator;

        switch (method) {
            case FLETCHER_RIEVES:
                return currentGradient.innerProduct(currentGradient) / previousGradient.innerProduct(previousGradient);
            case POLAK_RIBIERE:
                return currentGradient.innerProduct(currentGradient.subtract(previousGradient))
                        / previousGradient.innerProduct(previousGradient);
            case POLAK_RIBIERE_PLUS:
                return Math.max(currentGradient.innerProduct(currentGradient.subtract(previousGradient))
                                        / previousGradient.innerProduct(previousGradient), 0);
            case HESTENES_STIEFEL:
                gradientsDifference = currentGradient.subtract(previousGradient);
                return currentGradient.innerProduct(gradientsDifference)
                        / gradientsDifference.innerProduct(previousDirection);
            case FLETCHER_RIEVES_POLAK_RIBIERE:
                denominator = previousGradient.innerProduct(previousGradient);
                double betaFR = currentGradient.innerProduct(currentGradient) / denominator;
                double betaPR = currentGradient.innerProduct(currentGradient.subtract(previousGradient)) / denominator;
                if (betaPR < -betaFR) {
                    return -betaFR;
                } else if (betaPR > betaFR) {
                    return betaFR;
                } else {
                    return betaPR;
                }
            case DAI_YUAN:
                return currentGradient.innerProduct(currentGradient)
                        / currentGradient.subtract(previousGradient).innerProduct(previousDirection);
            case HAGER_ZHANG:
                gradientsDifference = currentGradient.subtract(previousGradient);
                denominator = gradientsDifference.innerProduct(previousDirection);
                Vector temporaryTerm = gradientsDifference.subtract(
                        previousDirection.multiply(2 * gradientsDifference.innerProduct(gradientsDifference)
                                                              / denominator)
                );
                return temporaryTerm.innerProduct(currentGradient) / denominator;
            default:
                throw new NotImplementedException();
        }
    }

    public Method getMethod() {
        return method;
    }

    public void setMethod(Method method) {
        this.method = method;
    }

    public RestartMethod getRestartMethod() {
        return restartMethod;
    }

    public void setRestartMethod(RestartMethod restartMethod) {
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

    /**
     * An enumeration of the currently supported nonlinear conjugate gradient optimization methods. They are very
     * similar to each other, but there are some significant differences that one should consider when choosing which
     * method to use. Here we provide a short description of some of those differences.
     * <br><br>
     * One weakness of the Fletcher-Rieves method is that when the algorithm generates a bad direction and a tiny step
     * size, then the next direction and the next step are also likely to be poor. The Polak-Ribiere method effectively
     * performs a restart when that happens. The same holds for the Polak-Ribiere+ and the Hestenes-Stiefel methods. The
     * Fletcher-Rieves-Polak-Ribiere method also deals well with that issue and at the same time retains the global
     * convergence properties of the Fletcher-Rieves method. The Fletcher-Rieves method, if used instead of one of the
     * other methods, it should be used along with some restart strategy in order to avoid that problem.
     * <br><br>
     * The Polak-Ribiere method can end up cycling infinitely without converging to a solution. In that case it might be
     * better to use the Polak-Ribiere+ method. Furthermore, we can prove global convergence results for the
     * Fletcher-Rieves, the Fletcher-Rieves-Polak-Ribiere, the Dai-Yuan and the Hager-Zhang methods, but we cannot prove
     * global convergence results for the Polak-Ribiere method. However, in practice the Polak-Ribiere+ method seems to
     * be the fastest one (and we can prove global convergence results for this method when used with certain line
     * search algorithms).
     * <br><br>
     * The Dai-Yuan method is based on the following paper: Y. Dai and Y. Yuan, A nonlinear conjugate gradient method
     * with a strong global convergence property, SIAM Journal on Optimization, 10 (1999), pp. 177&#45;182.
     * <br><br>
     * The Hager-Zhang method is based on the following paper: W. W. Hager and H. Zhang, A new conjugate gradient method
     * with guaranteed descent and an efficient line search, SIAM Journal on Optimization, 16 (2005), pp. 170&#45;192.
     */
    public enum Method {
        FLETCHER_RIEVES,
        POLAK_RIBIERE,
        POLAK_RIBIERE_PLUS,
        HESTENES_STIEFEL,
        FLETCHER_RIEVES_POLAK_RIBIERE,
        DAI_YUAN,
        HAGER_ZHANG
    }

    public enum RestartMethod {
        NO_RESTART,
        N_STEP,
        GRADIENTS_ORTHOGONALITY_CHECK
    }
}
