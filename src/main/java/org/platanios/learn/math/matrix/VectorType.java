package org.platanios.learn.math.matrix;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

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
        public DenseVector buildVector(int size, double value) {
            return new DenseVector(size, value);
        }

        /** {@inheritDoc} */
        @Override
        public DenseVector buildVector(double[] elements) {
            return new DenseVector(elements);
        }
    },
    SPARSE {
        /** {@inheritDoc} */
        @Override
        public Vector buildVector(int size, double value) {
            throw new NotImplementedException();
        }

        /** {@inheritDoc} */
        @Override
        public Vector buildVector(double[] elements) {
            throw new NotImplementedException();
        }
    };

    /**
     * Builds a vector of the corresponding type and of the given size and fills it with the provided value.
     *
     * @param   size    The size of the vector.
     * @param   value   The value with which to fill the vector.
     * @return          The new vector.
     */
    public abstract Vector buildVector(int size, double value);

    /**
     * Builds a vector of the corresponding type from a one-dimensional array.
     * TODO: Talk about how this method handles sparse vectors (i.e., does it threshold the values?).
     *
     * @param   elements    One-dimensional array of doubles.
     * @return              The new vector.
     */
    public abstract Vector buildVector(double[] elements);
}
