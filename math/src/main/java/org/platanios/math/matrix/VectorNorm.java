package org.platanios.math.matrix;

import org.platanios.math.MathUtilities;

import java.util.Collection;

/**
 * An enumeration of the supported norm operators for vectors.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum VectorNorm {
    /**
     * The \(L_1\) norm of this vector. Denoting a vector by \(\boldsymbol{x}\in\mathbb{R}^{n}\), its element at index
     * \(i\) by \(x_i\) and its \(L_1\) norm by \(\|\boldsymbol{x}\|_1\), we have that:
     * \[\|\boldsymbol{x}\|_1=\sum_{i=1}^n{\left|x_i\right|}.\]
     */
    L1 {
        /** {@inheritDoc} */
        @Override
        public double compute(double[] nonzeroValues) {
            double l1Norm = 0;
            for (double value : nonzeroValues) {
                l1Norm += Math.abs(value);
            }
            return l1Norm;
        }

        /** {@inheritDoc} */
        @Override
        public double compute(Collection<Double> nonzeroValues) {
            double l1Norm = 0;
            for (double value : nonzeroValues) {
                l1Norm += Math.abs(value);
            }
            return l1Norm;
        }
    },
    /**
     * The \(L_2\) norm of this vector. Denoting a vector by \(\boldsymbol{x}\in\mathbb{R}^{n}\), its element at index
     * \(i\) by \(x_i\) and its \(L_2\) norm by \(\|\boldsymbol{x}\|_2\), we have that:
     * \[\|\boldsymbol{x}\|_2=\sqrt{\sum_{i=1}^n{x_i^2}}.\]
     * This implementation attempts to avoid numerical underflow or overflow, but is slower than {@link #L2_FAST} that
     * does not.
     */
    L2 {
        /** {@inheritDoc} */
        @Override
        public double compute(double[] nonzeroValues) {
            double l2Norm = 0;
            for (double value : nonzeroValues) {
                l2Norm = MathUtilities.computeHypotenuse(l2Norm, value);
            }
            return l2Norm;
        }

        /** {@inheritDoc} */
        @Override
        public double compute(Collection<Double> nonzeroValues) {
            double l2Norm = 0;
            for (double value : nonzeroValues) {
                l2Norm = MathUtilities.computeHypotenuse(l2Norm, value);
            }
            return l2Norm;
        }
    },
    /**
     * The \(L_2\) norm of this vector. Denoting a vector by \(\boldsymbol{x}\in\mathbb{R}^{n}\), its element at index
     * \(i\) by \(x_i\) and its \(L_2\) norm by \(\|\boldsymbol{x}\|_2\), we have that:
     * \[\|\boldsymbol{x}\|_2=\sqrt{\sum_{i=1}^n{x_i^2}}.\]
     * This implementation does not attempt to avoid numerical underflow or overflow, but is faster than {@link #L2}
     * that does.
     */
    L2_FAST {
        /** {@inheritDoc} */
        @Override
        public double compute(double[] nonzeroValues) {
            double l2Norm = 0;
            for (double value : nonzeroValues) {
                l2Norm += value * value;
            }
            return Math.sqrt(l2Norm);
        }

        /** {@inheritDoc} */
        @Override
        public double compute(Collection<Double> nonzeroValues) {
            double l2Norm = 0;
            for (double value : nonzeroValues) {
                l2Norm += value * value;
            }
            return Math.sqrt(l2Norm);
        }
    },
    L2_SQUARED {
        /** {@inheritDoc} */
        @Override
        public double compute(double[] nonzeroValues) {
            double l2NormSquared = 0;
            for (double value : nonzeroValues) {
                l2NormSquared += value * value;
            }
            return l2NormSquared;
        }

        /** {@inheritDoc} */
        @Override
        public double compute(Collection<Double> nonzeroValues) {
            double l2NormSquared = 0;
            for (double value : nonzeroValues) {
                l2NormSquared += value * value;
            }
            return l2NormSquared;
        }
    },
    /**
     * The \(L_\infty\) norm of this vector. Denoting a vector by \(\boldsymbol{x}\in\mathbb{R}^{n}\), its element at
     * index \(i\) by \(x_i\) and its \(L_\infty\) norm by \(\|\boldsymbol{x}\|_\infty\), we have that:
     * \[\|\boldsymbol{x}\|_\infty=\max_{1\leq i\leq n}{\left|x_i\right|}.\]
     */
    LINFINITY {
        /** {@inheritDoc} */
        @Override
        public double compute(double[] nonzeroValues) {
            double lInfinityNorm = 0;
            for (double value : nonzeroValues) {
                lInfinityNorm = Math.max(lInfinityNorm, value);
            }
            return lInfinityNorm;
        }

        /** {@inheritDoc} */
        @Override
        public double compute(Collection<Double> nonzeroValues) {
            double lInfinityNorm = 0;
            for (double value : nonzeroValues) {
                lInfinityNorm = Math.max(lInfinityNorm, value);
            }
            return lInfinityNorm;
        }
    };

    /**
     * Computes the specified norm of a vector. Only the nonzero values of the vector are required in order to compute
     * its norm.
     *
     * @param   nonzeroValues   The nonzero values of the vector provided as an array.
     * @return                  The specified norm of this vector.
     */
    public abstract double compute(double[] nonzeroValues);

    /**
     * Computes the specified norm of a vector. Only the nonzero values of the vector are required in order to compute
     * its norm.
     *
     * @param   nonzeroValues   The nonzero values of the vector provided as a collection.
     * @return                  The specified norm of this vector.
     */
    public abstract double compute(Collection<Double> nonzeroValues);
}
