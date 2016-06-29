package makina.math.matrix;

import java.io.IOException;
import java.io.InputStream;

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
        public DenseVector buildVector(InputStream inputStream, boolean includeType) throws IOException {
            return DenseVector.read(inputStream, includeType);
        }
    },
    SPARSE {
        /** {@inheritDoc} */
        @Override
        public SparseVector buildVector(int size) {
            return new SparseVector(size);
        }

        /** {@inheritDoc} */
        @Override
        public SparseVector buildVector(InputStream inputStream, boolean includeType) throws IOException {
            return SparseVector.read(inputStream, includeType);
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
        public HashVector buildVector(InputStream inputStream, boolean includeType) throws IOException {
            return HashVector.read(inputStream, includeType);
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
     * of the stream must have been written using the write(InputStream, boolean) function of the corresponding vector
     * class in order to be compatible with this constructor. If the contents are not compatible, then an
     * {@link IOException} might be thrown, or the constructed vector might be corrupted in some way.
     *
     * @param   inputStream The input stream to read the contents of this vector from.
     * @param   includeType Boolean computeValue indicating whether the type of the vector is to also be read from the input
     *                      stream.
     * @return              The new vector.
     * @throws  IOException
     */
    public abstract Vector buildVector(InputStream inputStream, boolean includeType) throws IOException;
}
