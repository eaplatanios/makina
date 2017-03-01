package makina.optimization;

import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.math.matrix.VectorNorm;
import makina.optimization.linesearch.StepSizeInitializationMethod;
import makina.optimization.function.AbstractFunction;
import makina.optimization.linesearch.StrongWolfeInterpolationLineSearch;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class QuasiNewtonSolver extends AbstractLineSearchSolver {
    private final Matrix identityMatrix;
    private final Method method;
    private final int m;
    
    private final Matrix initialHessian;
    private Matrix currentH;
    private Matrix previousH;
    Vector[] s;
    Vector[] y;
    private Vector initialHessianInverseDiagonal = Vectors.dense(currentPoint.size(), 1);

    private double symmetricRankOneSkippingParameter = 1e-8;

    public final Vector currentPoint() {
    	return currentPoint;
    }
    public final double currentValue() {
    	return currentObjectiveValue;
    }
    public final Matrix currentH() {
    	return this.currentH;
    }

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends AbstractLineSearchSolver.AbstractBuilder<T> {
        private Method method = Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO;
        private int m = 1;
        private Matrix initialHessian = Matrix.identity(initialPoint.size());
        private double symmetricRankOneSkippingParameter = 1e-8;

        public AbstractBuilder(AbstractFunction objective, Vector initialPoint) {
            super(objective, initialPoint);
            lineSearch = new StrongWolfeInterpolationLineSearch(objective, 1e-4, 0.9, 1000);
            ((StrongWolfeInterpolationLineSearch) lineSearch)
                    .setStepSizeInitializationMethod(StepSizeInitializationMethod.UNIT);
        }

        public T method(Method method) {
            this.method = method;

            if (method != Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO) {
                m = 1;
            } else if (m == 1) {
                m = 10;
            }

            return self();
        }

        public T m(int m) {
            if (method != Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO) {
                m = 1;
            }

            this.m = m;
            return self();
        }
        
        public T initialHessian(final Matrix initialHessian) {
        	this.initialHessian = initialHessian;
        	return self();
        }

        public T symmetricRankOneSkippingParameter(double symmetricRankOneSkippingParameter) {
            this.symmetricRankOneSkippingParameter = symmetricRankOneSkippingParameter;
            return self();
        }

        public QuasiNewtonSolver build() {
            return new QuasiNewtonSolver(this);
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

    /**
     * BROYDEN_FLETCHER_GOLDFARB_SHANNO method chosen by default. The default value used for the memory parameter,
     * {@code m}, is 10 for L-BFGS (for all other methods it is 1, since they store the whole approximation matrix and
     * do not need to store any previous vectors to re-construct it using limited memory).
     *
     */
    private QuasiNewtonSolver(AbstractBuilder<?> builder) {
        super(builder);
        method = builder.method;
        m = builder.m;
        initialHessian = builder.initialHessian;
        symmetricRankOneSkippingParameter = builder.symmetricRankOneSkippingParameter;
        identityMatrix = Matrix.identity(builder.initialPoint.size());
        currentH = initialHessian.copy();
        s = new Vector[m];
        y = new Vector[m];
    }

    @Override
    public void updateDirection() {
        updateStoredVectors();
        method.updateDirection(this);
    }

    @Override
    public void updatePoint() {
        currentPoint = previousPoint.add(currentDirection.mult(currentStepSize));
    }

    /** m != 1 only for the LBFGS method. */
    private void updateStoredVectors() {
        for (int i = m - 1; i > 0; i--) {
            s[i] = s[i-1];
            y[i] = y[i-1];
        }
        s[0] = currentPoint.sub(previousPoint);
        y[0] = currentGradient.sub(previousGradient);
    }

    /**
     * For all the methods we use a simple trick to initialize the Hessian inverse approximation. We use a method that
     * tries to make the size of the Hessian matrix similar to the size of the actual Hessian matrix inverse (the
     * method that we use attempts to estimate the size of the true Hessian matrix along the most recent search
     * direction. This choice helps to ensure that the search direction is well scaled and as a result the step length
     * value 1 is accepted in most iterations.
     */
    public enum Method {
        /** The Davidon–Fletcher–Powell algorithm. This algorithm is less effective than BROYDEN_FLETCHER_GOLDFARB_SHANNO at correcting bad Hessian
         * approximations. Both this method and the BROYDEN_FLETCHER_GOLDFARB_SHANNO method preserve the positive-definiteness of the Hessian matrix. */
        DAVIDON_FLETCHER_POWELL {
            @Override
            protected void updateDirection(QuasiNewtonSolver solver) {
                if (solver.currentIteration > 0) {
                    updatePreviousH(solver);
                    solver.currentH = solver.previousH.subtract(solver.previousH.multiply(
                            solver.y[0].mult(1 / solver.y[0].transMult(solver.previousH).inner(solver.y[0]))
                                    .outer(solver.y[0])
                                    .multiply(solver.previousH)
                    )).add(solver.s[0].mult(1 / solver.y[0].inner(solver.s[0])).outer(solver.s[0]));
                }
                solver.currentDirection = solver.currentH.multiply(solver.currentGradient).mult(-1);
            }
        },
        /** The Broyden–Fletcher–Goldfarb–Shanno algorithm. This algorithm is very good at correcting bad Hessian
         * approximations. */
        BROYDEN_FLETCHER_GOLDFARB_SHANNO {
            @Override
            protected void updateDirection(QuasiNewtonSolver solver) {
                if (solver.currentIteration > 0) {
                    updatePreviousH(solver);
                    double rho = 1 / solver.y[0].inner(solver.s[0]);
                    solver.currentH = solver.identityMatrix
                            .subtract(solver.s[0].mult(rho).outer(solver.y[0]))
                            .multiply(solver.previousH)
                            .multiply(solver.identityMatrix
                                              .subtract(solver.y[0].mult(rho).outer(solver.s[0])))
                            .add(solver.s[0].mult(rho).outer(solver.s[0]));
                }
                solver.currentDirection = solver.currentH.multiply(solver.currentGradient).mult(-1);
            }
        },
        LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO {
            @Override
            protected void updateDirection(QuasiNewtonSolver solver) {
                if (solver.currentIteration > 0) {
                    solver.initialHessianInverseDiagonal =
                            Vectors.dense(solver.currentPoint.size(), 1)
                                    .mult(solver.s[0].inner(solver.y[0])
                                                  / solver.y[0].inner(solver.y[0]));
                }
                solver.currentDirection =
                        approximateHessianInverseVectorProduct(solver, solver.currentGradient).mult(-1);
            }

            private Vector approximateHessianInverseVectorProduct(QuasiNewtonSolver solver, Vector q) {
                double[] a = new double[solver.m];
                double[] rho = new double[solver.m];
                for (int i = 0; i < Math.min(solver.m, solver.currentIteration); i++) {
                    rho[i] = 1 / solver.y[i].inner(solver.s[i]);
                    a[i] = rho[i] * solver.s[i].inner(q);
                    q = q.sub(solver.y[i].mult(a[i]));
                }
                Vector result = q.multElementwise(solver.initialHessianInverseDiagonal);
                double temp0 = solver.y[0].inner(result);
                double temp1 = rho[0] * temp0;
                double temp2 = a[0] - temp1;
                Vector temp3 = solver.s[0].mult(temp2);
                Vector temp4 = (result.mult(-1.0)).add(temp3);
                for (int i = Math.min(solver.m, solver.currentIteration) - 1; i >= 0; i--) {
                    result = result.add(solver.s[i].mult(a[i] - rho[i] * solver.y[i].inner(result)));
                }
                return result;
            }
        },
        /** The Symmetric-Rank-1 algorithm. This method may produce indefinite Hessian approximations. Furthermore, the
         * basic SYMMETRIC_RANK_ONE method may break down and that is why here it has been implemented with a skipping method to help
         * prevent such cases. */
        SYMMETRIC_RANK_ONE {
            @Override
            protected void updateDirection(QuasiNewtonSolver solver) {
                if (solver.currentIteration > 0) {
                    updatePreviousH(solver);
                    Vector tempVector = solver.s[0].sub(solver.previousH.multiply(solver.y[0]));
                    if (Math.abs(tempVector.inner(solver.y[0]))
                            >= solver.symmetricRankOneSkippingParameter
                            * solver.y[0].norm(VectorNorm.L2)
                            *  tempVector.norm(VectorNorm.L2)) {
                        solver.currentH = solver.previousH.add(
                                tempVector.mult(1 / tempVector.inner(solver.y[0])).outer(tempVector)
                        );
                    } else {
                        solver.currentH = solver.previousH;
                    }
                }
                solver.currentDirection = solver.currentH.multiply(solver.currentGradient).mult(-1);
            }
        },
        BROYDEN {
            @Override
            protected void updateDirection(QuasiNewtonSolver solver) {
                if (solver.currentIteration > 0) {
                    updatePreviousH(solver);
                    solver.currentH = solver.previousH.add(
                            solver.s[0].sub(solver.previousH.multiply(solver.y[0]))
                                    .outer(solver.s[0].transMult(solver.previousH))
                                    .multiply(1 / solver.s[0].transMult(solver.previousH).inner(solver.y[0]))
                    );
                }
                solver.currentDirection = solver.currentH.multiply(solver.currentGradient).mult(-1);
            }
        };

        protected abstract void updateDirection(QuasiNewtonSolver solver);

        /**
         * Used by all methods except from the limited memory Broyden-Fletcher-Goldfarb-Shanno method.
         *
         * @param   solver  The actual solver object whose {@code previousH} variable is updated.
         */
        private static void updatePreviousH(QuasiNewtonSolver solver) {
            if (solver.currentIteration == 1) {
                solver.previousH = solver.currentH
                        .multiply(solver.y[0].inner(solver.s[0])
                                          / solver.y[0].inner(solver.y[0]));
            } else {
                solver.previousH = solver.currentH;
            }
        }
    }
}
