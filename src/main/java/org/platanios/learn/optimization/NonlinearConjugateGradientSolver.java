package org.platanios.learn.optimization;

import org.platanios.learn.math.matrix.VectorNorm;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.math.matrix.Vector;

/**
 * These solvers are good for large scale problems and, on certain applications, competitive with limited-memory
 * quasi-Newton methods as well.
 *
 * @author Emmanouil Antonios Platanios
 */
public final class NonlinearConjugateGradientSolver extends AbstractLineSearchSolver {
    /** Default value: If quadratic or linear function it is ExactLineSearch, otherwise it is StrongWolfeLineSearch
     * with CONSERVE_FIRST_ORDER_CHANGE for the step size initialization method. */
    private final Method method;
    private final RestartMethod restartMethod;
    private final double gradientsOrthogonalityCheckThreshold;

    // The following variables are used locally within iteration but are initialized here in order to make the code more
    // clear.
    double beta;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractLineSearchSolver.AbstractBuilder<T> {
        private Method method = Method.POLAK_RIBIERE_PLUS;
        private RestartMethod restartMethod = RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK;
        private double gradientsOrthogonalityCheckThreshold = 0.1;

        public AbstractBuilder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
            checkForPointConvergence = false;
            checkForObjectiveConvergence = false;
        }

        public T method(Method method) {
            this.method = method;
            return self();
        }

        public T restartMethod(RestartMethod restartMethod) {
            this.restartMethod = restartMethod;
            return self();
        }

        public T gradientsOrthogonalityCheckThreshold(double gradientsOrthogonalityCheckThreshold) {
            this.gradientsOrthogonalityCheckThreshold = gradientsOrthogonalityCheckThreshold;
            return self();
        }

        public NonlinearConjugateGradientSolver build() {
            return new NonlinearConjugateGradientSolver(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(AbstractFunction objective,
                       Vector initialPoint) {
            super(objective, initialPoint);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private NonlinearConjugateGradientSolver(AbstractBuilder<?> builder) {
        super(builder);
        lineSearch = builder.lineSearch;
        method = builder.method;
        restartMethod = builder.restartMethod;
        gradientsOrthogonalityCheckThreshold = builder.gradientsOrthogonalityCheckThreshold;
        currentDirection = currentGradient.mult(-1);
    }

    @Override
    public void updateDirection() {
        beta = checkForRestart() ? 0 : method.computeBeta(this);
        currentDirection = currentGradient.mult(-1).add(previousDirection.mult(beta));
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.mult(currentStepSize));
    }

    private boolean checkForRestart() {
        return restartMethod.checkForRestart(this);
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
        FLETCHER_RIEVES {
            @Override
            protected double computeBeta(NonlinearConjugateGradientSolver solver) {
                return solver.currentGradient.inner(solver.currentGradient)
                        / solver.previousGradient.inner(solver.previousGradient);
            }
        },
        POLAK_RIBIERE {
            @Override
            protected double computeBeta(NonlinearConjugateGradientSolver solver) {
                return solver.currentGradient.inner(solver.currentGradient.sub(solver.previousGradient))
                        / solver.previousGradient.inner(solver.previousGradient);
            }
        },
        POLAK_RIBIERE_PLUS {
            @Override
            protected double computeBeta(NonlinearConjugateGradientSolver solver) {
                return Math.max(
                        solver.currentGradient.inner(solver.currentGradient.sub(solver.previousGradient))
                                / solver.previousGradient.inner(solver.previousGradient), 0);
            }
        },
        HESTENES_STIEFEL {
            @Override
            protected double computeBeta(NonlinearConjugateGradientSolver solver) {
                Vector gradientsDifference = solver.currentGradient.sub(solver.previousGradient);
                if (gradientsDifference.norm(VectorNorm.L2) != 0) {
                    return solver.currentGradient.inner(gradientsDifference)
                            / gradientsDifference.inner(solver.previousDirection);
                } else {
                    return 0;
                }
            }
        },
        FLETCHER_RIEVES_POLAK_RIBIERE {
            @Override
            protected double computeBeta(NonlinearConjugateGradientSolver solver) {
                double denominator = solver.previousGradient.inner(solver.previousGradient);
                double betaFR = solver.currentGradient.inner(solver.currentGradient) / denominator;
                double betaPR = solver.currentGradient
                        .inner(solver.currentGradient.sub(solver.previousGradient)) / denominator;
                if (betaPR < -betaFR) {
                    return -betaFR;
                } else if (betaPR > betaFR) {
                    return betaFR;
                } else {
                    return betaPR;
                }
            }
        },
        DAI_YUAN {
            @Override
            protected double computeBeta(NonlinearConjugateGradientSolver solver) {
                Vector gradientsDifference = solver.currentGradient.sub(solver.previousGradient);
                if (gradientsDifference.norm(VectorNorm.L2) != 0) {
                    return solver.currentGradient.inner(solver.currentGradient)
                            / gradientsDifference.inner(solver.previousDirection);
                } else {
                    return 0;
                }
            }
        },
        HAGER_ZHANG {
            @Override
            protected double computeBeta(NonlinearConjugateGradientSolver solver) {
                Vector gradientsDifference = solver.currentGradient.sub(solver.previousGradient);
                if (gradientsDifference.norm(VectorNorm.L2) != 0) {
                    double denominator = gradientsDifference.inner(solver.previousDirection);
                    Vector temporaryTerm = gradientsDifference.sub(
                            solver.previousDirection.mult(2 * gradientsDifference.inner(gradientsDifference)
                                                                  / denominator)
                    );
                    return temporaryTerm.inner(solver.currentGradient) / denominator;
                } else {
                    return 0;
                }
            }
        };

        protected abstract double computeBeta(NonlinearConjugateGradientSolver solver);
    }

    public enum RestartMethod {
        NO_RESTART {
            @Override
            protected boolean checkForRestart(NonlinearConjugateGradientSolver solver) {
                return false;
            }
        },
        N_STEP {
            @Override
            protected boolean checkForRestart(NonlinearConjugateGradientSolver solver) {
                return solver.currentIteration % solver.currentPoint.size() == 0;
            }
        },
        GRADIENTS_ORTHOGONALITY_CHECK {
            @Override
            protected boolean checkForRestart(NonlinearConjugateGradientSolver solver) {
                return Math.abs(solver.currentGradient.inner(solver.previousGradient))
                        / solver.currentGradient.inner(solver.currentGradient)
                        >= solver.gradientsOrthogonalityCheckThreshold;
            }
        };

        protected abstract boolean checkForRestart(NonlinearConjugateGradientSolver solver);
    }
}
