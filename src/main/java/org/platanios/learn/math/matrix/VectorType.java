package org.platanios.learn.math.matrix;

import java.io.IOException;
import java.io.ObjectInputStream;

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

        /** {@inheritDoc} */
        @Override
        public DenseVector buildVector(ObjectInputStream inputStream) throws IOException {
            return new DenseVector(inputStream);
        }
    },
    SPARSE {
        /** {@inheritDoc} */
        @Override
        public SparseVector buildVector(int size) {
            return new SparseVector.Builder(size).build();
        }

        /** {@inheritDoc} */
        @Override
        public SparseVector buildVector(ObjectInputStream inputStream) throws IOException {
            return new SparseVector.Builder(inputStream).build();
        }
    },
    HASH {
        /** {@inheritDoc} */
        @Override
        public HashVector buildVector(int size) {
            return new HashVector(size);
        }

        /** {@inheritDoc} */
        @Override
        public HashVector buildVector(ObjectInputStream inputStream) throws IOException {
            return new HashVector(inputStream);
        }
    };

    /**
     * Builds a vector of the corresponding type and of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @return          The new vector.
     */
    public abstract Vector buildVector(int size);

    /**
     * Builds a vector of the corresponding type from the contents of the provided input stream. Note that the contents
     * of the stream must have been written using the writeToStream(java.io.ObjectOutputStream) function of the
     * corresponding vector class in order to be compatible with this constructor. If the contents are not compatible,
     * then an {@link java.io.IOException} might be thrown, or the constructed vector might be corrupted in some way.
     *
     * @param   inputStream The input stream to read the contents of this vector from.
     * @return              The new vector.
     * @throws  IOException
     */
    public abstract Vector buildVector(ObjectInputStream inputStream) throws IOException;
}
