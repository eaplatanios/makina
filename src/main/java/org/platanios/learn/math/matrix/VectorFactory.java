package org.platanios.learn.math.matrix;

import cern.colt.map.OpenIntDoubleHashMap;

import java.io.IOException;
import java.io.ObjectInputStream;

/**
 * This class provides several static methods to build vectors of different types and initialize them in various ways.
 *
 * @author Emmanouil Antonios Platanios
 */
public class VectorFactory {
    /**
     * Builds a vector of the given size and type and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @param   type    The type of vector to build.
     * @return          The new vector.
     */
    public static Vector build(int size, VectorType type) {
        return type.buildVector(size);
    }

    /**
     * Builds a vector of the given type from the contents of the provided input stream. Note that the contents of the
     * stream must have been written using the writeToStream(java.io.ObjectOutputStream) function of the corresponding
     * vector class in order to be compatible with this constructor. If the contents are not compatible, then an
     * {@link java.io.IOException} might be thrown, or the constructed vector might be corrupted in some way.
     *
     * @param   inputStream The input stream to read the contents of this vector from.
     * @param   type        The type of vector to build.
     * @return              The new vector.
     * @throws  IOException
     */
    public static Vector build(ObjectInputStream inputStream, VectorType type) throws IOException {
        return type.buildVector(inputStream);
    }

    /**
     * Builds a dense vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @return          The new vector.
     */
    public static DenseVector buildDense(int size) {
        return new DenseVector(size);
    }

    /**
     * Builds a dense vector of the given size and fills it with the provided value.
     *
     * @param   size    The size of the vector.
     * @param   value   The value with which to fill the vector.
     * @return          The new vector.
     */
    public static DenseVector buildDense(int size, double value) {
        return new DenseVector(size, value);
    }

    /**
     * Builds a dense vector from a one-dimensional array.
     *
     * @param   elements    One-dimensional array of doubles.
     * @return              The new vector.
     */
    public static DenseVector buildDense(double[] elements) {
        return new DenseVector(elements);
    }

    /**
     * Constructs a dense vector from the contents of the provided input stream. Note that the contents of the stream
     * must have been written using the
     * {@link org.platanios.learn.math.matrix.DenseVector#writeToStream(java.io.ObjectOutputStream)} function of this
     * class in order to be compatible with this constructor. If the contents are not compatible, then an
     * {@link java.io.IOException} might be thrown, or the constructed vector might be corrupted in some way.
     *
     * @param   inputStream The input stream to read the contents of this vector from.
     * @return              The new vector.
     * @throws  IOException
     */
    public static DenseVector buildDense(ObjectInputStream inputStream) throws IOException {
        return new DenseVector(inputStream);
    }

    /**
     * Builds a sparse vector of the given size containing only zeros.
     *
     * @param   size        The size of the vector.
     * @return              The new vector.
     */
    public static SparseVector buildSparse(int size) {
        return new SparseVector(size);
    }

    /**
     * Builds a sparse vector of the given size from a hash map.
     *
     * @param   size        The size of the vector.
     * @param   elements    Hash map containing the indexes of elements as keys and the values of the corresponding
     *                      elements as values.
     * @return              The new vector.
     */
    public static SparseVector buildSparse(int size, OpenIntDoubleHashMap elements) {
        return new SparseVector(size, elements);
    }

    /**
     * Constructs a sparse vector from the contents of the provided input stream. Note that the contents of the stream
     * must have been written using the
     * {@link org.platanios.learn.math.matrix.SparseVector#writeToStream(java.io.ObjectOutputStream)} function of this
     * class in order to be compatible with this constructor. If the contents are not compatible, then an
     * {@link java.io.IOException} might be thrown, or the constructed vector might be corrupted in some way.
     *
     * @param   inputStream The input stream to read the contents of this vector from.
     * @return              The new vector.
     * @throws  IOException
     */
    public static SparseVector buildSparse(ObjectInputStream inputStream) throws IOException {
        return new SparseVector(inputStream);
    }
}
