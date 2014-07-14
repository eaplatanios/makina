package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.optimization.function.AbstractFunction;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * These solvers are good for large scale problems and, on certain applications, competitive with limited-memory
 * quasi-Newton methods as well.
 *
 * @author Emmanouil Antonios Platanios
 */
public class NonlinearConjugateGradientSolver extends AbstractLineSearchSolver {
    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private final Method method;
    private final RestartMethod restartMethod;
    private final double gradientsOrthogonalityCheckThreshold;

    // The following variables are used locally within iteration but are initialized here in order to make the code more
    // clear.
    double beta;

    public static class Builder extends AbstractLineSearchSolver.Builder<NonlinearConjugateGradientSolver> {
        private Method method = Method.POLAK_RIBIERE_PLUS;
        private RestartMethod restartMethod = RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK;
        private double gradientsOrthogonalityCheckThreshold = 0.1;

        public Builder(AbstractFunction objective, double[] initialPoint) {
            super(objective, initialPoint);
        }

        public Builder method(Method method) {
            this.method = method;
            return this;
        }

        public Builder restartMethod(RestartMethod restartMethod) {
            this.restartMethod = restartMethod;
            return this;
        }

        public Builder gradientsOrthogonalityCheckThreshold(double gradientsOrthogonalityCheckThreshold) {
            this.gradientsOrthogonalityCheckThreshold = gradientsOrthogonalityCheckThreshold;
            return this;
        }

        public NonlinearConjugateGradientSolver build() {
            return new NonlinearConjugateGradientSolver(this);
        }
    }

    public NonlinearConjugateGradientSolver(Builder builder) {
        super(builder);
        this.lineSearch = builder.lineSearch;
        this.method = builder.method;
        this.restartMethod = builder.restartMethod;
        this.gradientsOrthogonalityCheckThreshold = builder.gradientsOrthogonalityCheckThreshold;
        setCheckForPointConvergence(false);
        setCheckForObjectiveConvergence(false);
        currentDirection = currentGradient.multiply(-1);
    }

    @Override
    public void updateDirection() {
        beta = checkForRestart() ? 0 : computeBeta();
        currentDirection = currentGradient.multiply(-1).add(previousDirection.multiply(beta));
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.multiply(currentStepSize));
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
                if (gradientsDifference.computeL2Norm() != 0) {
                    return currentGradient.innerProduct(gradientsDifference)
                            / gradientsDifference.innerProduct(previousDirection);
                } else {
                    return 0;
                }
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
                gradientsDifference = currentGradient.subtract(previousGradient);
                if (gradientsDifference.computeL2Norm() != 0) {
                    return currentGradient.innerProduct(currentGradient)
                            / gradientsDifference.innerProduct(previousDirection);
                } else {
                    return 0;
                }
            case HAGER_ZHANG:
                gradientsDifference = currentGradient.subtract(previousGradient);
                if (gradientsDifference.computeL2Norm() != 0) {
                    denominator = gradientsDifference.innerProduct(previousDirection);
                    Vector temporaryTerm = gradientsDifference.subtract(
                            previousDirection.multiply(2 * gradientsDifference.innerProduct(gradientsDifference)
                                                               / denominator)
                    );
                    return temporaryTerm.innerProduct(currentGradient) / denominator;
                } else {
                    return 0;
                }
            default:
                throw new NotImplementedException();
        }
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
