package org.platanios.learn.math.matrix;

/**
 * This enumeration contains the different types of vectors that are supported. Each type also contains methods that can
 * be called to build vectors of the corresponding type.
 *
 * @author Emmanouil Antonios Platanios
 */
public enum VectorType {
    DENSE {
        /** {@inheritDoc} */
        @Override
        public DenseVector buildVector(int size) {
            return new DenseVector(size);
        }

        /**
         * Builds a dense vector of the given size and fills it with the provided value.
         *
         * @param   size    The size of the vector.
         * @param   value   The value with which to fill the vector.
         * @return          The new vector.
         */
        public DenseVector buildVector(int size, double value) {
            return new DenseVector(size, value);
        }

        /**
         * Builds a dense vector and fills it with the values of the provided one-dimensional array.
         *
         * @param   elements    One-dimensional array of doubles.
         * @return              The new vector.
         */
        public DenseVector buildVector(double[] elements) {
            return new DenseVector(elements);
        }
    },
    SPARSE {
        /** {@inheritDoc} */
        @Override
        public SparseVector buildVector(int size) {
            return new SparseVector(size);
        }
    };

    /**
     * Builds a vector of the corresponding type and of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @return          The new vector.
     */
    public abstract Vector buildVector(int size);
}
