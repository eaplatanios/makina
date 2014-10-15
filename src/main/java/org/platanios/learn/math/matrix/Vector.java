package org.platanios.learn.math.matrix;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.function.Function;

/**
 * Interface for classes representing vectors and supporting operations related to vectors.
 *
 * TODO: Allow transposing all matrix arguments.
 *
 * @author Emmanouil Antonios Platanios
 */
public abstract class Vector {
    private static final long serialVersionUID = -6542607523957470903L;

    /** The threshold value for elements to be considered equal to zero when counting the number of non-zero elements of
     * this vector (i.e., in method {@link #cardinality()}) and when handling sparse vectors. */
    protected final double epsilon = Math.sqrt(Double.MIN_VALUE);

    /**
     * Gets the type of this vector (i.e., dense, sparse, etc.).
     *
     * @return  The type of this vector.
     */
    public abstract VectorType type();

    /**
     * Copies this vector.
     *
     * @return  A copy of this vector.
     */
    public abstract Vector copy();

    /**
     * Gets a dense array representation of this vector. This array is completely separate from the inner representation
     * used by the vector implementation.
     *
     * @return  A dense array representation of this vector.
     */
    public abstract double[] getDenseArray();

    /**
     * Gets the dimension of this vector.
     *
     * @return  The dimension of this vector.
     */
    public abstract int size();

    /**
     * Gets the cardinality of this vector. The cardinality of a vector is the number of nonzero elements it contains.
     *
     * @return  The cardinality of this vector.
     */
    public abstract int cardinality();

    /**
     * Gets the value of the vector element at the provided index.
     *
     * @param   index   The index of the element.
     * @return          The value of the element at the provided index.
     *
     * @throws  java.lang.IllegalArgumentException  The provided index must be between 0 (inclusive) and the size of the
     *                                              vector (exclusive).
     */
    public abstract double get(int index);

    /**
     * Gets a sub-vector of this vector.
     *
     * @param   initialIndex    The initial index.
     * @param   finalIndex      The final index (inclusive).
     * @return                  The sub-vector corresponding to the provided indexes.
     *
     * @throws  java.lang.IllegalArgumentException  The provided indexes must be between 0 (inclusive) and the size of
     *                                              the vector (exclusive) and the initial index must be smaller or
     *                                              equal to the final index.
     */
    public abstract Vector get(int initialIndex, int finalIndex);

    /**
     * Gets a sub-vector of this vector.
     *
     * @param   indexes The indexes of the elements of this vector to be included in the returned sub-vector.
     * @return          The sub-vector corresponding to the provided indexes.
     *
     * @throws  java.lang.IllegalArgumentException  The provided indexes must be between 0 (inclusive) and the size of
     *                                              the vector (exclusive).
     */
    public abstract Vector get(int[] indexes);

    /**
     * Sets the value of the vector element at the provided index to the provided value.
     *
     * @param   index   The index of the element.
     * @param   value   The value to which to set the element at the provided index.
     *
     * @throws  java.lang.IllegalArgumentException  The provided index must be between 0 (inclusive) and the size of the
     *                                              vector (exclusive).
     */
    public abstract void set(int index, double value);

    /**
     * Sets a sub-vector of this vector to the provided vector values.
     *
     * @param   initialIndex    The initial index.
     * @param   finalIndex      The final index (inclusive).
     * @param   vector          The vector to whose values we set the values of the specified sub-vector of this vector.
     *
     * @throws  java.lang.IllegalArgumentException  The provided indexes must be between 0 (inclusive) and the size of
     *                                              the vector (exclusive) and the initial index must be smaller or
     *                                              equal to the final index.
     */
    public abstract void set(int initialIndex, int finalIndex, Vector vector);

    /**
     * Sets a sub-vector of this matrix to the provided vector values.
     *
     * @param   indexes     The indexes of the elements of this vector to be changed to values of the elements of the
     *                      provided sub-vector.
     * @param   vector      The vector to whose values we set the values of the specified sub-vector of this vector.
     *
     * @throws  java.lang.IllegalArgumentException  The provided indexes must be between 0 (inclusive) and the size of
     *                                              the vector (exclusive).
     */
    public abstract void set(int[] indexes, Vector vector);

    /**
     * Sets the value of all of the vector elements to the provided value.
     *
     * @param   value   The value to which to set the elements of this vector.
     */
    public abstract void setAll(double value);

    /**
     * Gets the maximum value of all elements in this vector.
     *
     * @return  The maximum value of all elements in this vector.
     */
    public abstract double max();

    /**
     * Gets the minimum value of all elements in this vector.
     *
     * @return  The minimum value of all elements in this vector.
     */
    public abstract double min();

    /**
     * Computes and returns the sum of all elements in this vector.
     *
     * @return  The sum of all elements in this vector.
     */
    public abstract double sum();

    /**
     * Computes the specified norm of this vector.
     *
     * @return  The specified norm of this vector.
     */
    public abstract double norm(VectorNorm normType);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and returns it in a new
     * vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @return              A new vector holding the result of the operation.
     */
    public abstract Vector map(Function<Double, Double> function);

    /**
     * Adds a scalar to all entries of the current vector and returns the result in a new vector.
     *
     * @param   scalar  The scalar to add to entries of the current vector.
     * @return          A new vector holding the result of the addition.
     */
    public abstract Vector add(double scalar);

    /**
     * Adds a scalar to all entries of the current vector and replaces the current vector with the result.
     *
     * @param   scalar  The scalar to add to entries of the current vector.
     */
    public abstract Vector addInPlace(double scalar);

    /**
     * Adds another vector to the current vector and returns the result in a new vector.
     *
     * @param   vector  The vector to add to the current vector.
     * @return          A new vector holding the result of the addition.
     */
    public abstract Vector add(Vector vector);

    /**
     * Adds another vector to the current vector and replaces the current vector with the result.
     *
     * @param   vector  The vector to add to the current vector.
     */
    public abstract Vector addInPlace(Vector vector);

    /**
     * Subtracts a scalar from all entries of the current vector and returns the result in a new vector.
     *
     * @param   scalar  The scalar to subtract from all entries of the current vector.
     * @return          A new vector holding the result of the subtraction.
     */
    public abstract Vector sub(double scalar);

    /**
     * Subtracts a scalar from all entries of the current vector and replaces the current vector with the result.
     *
     * @param   scalar  The scalar to subtract from all entries of the current vector.
     */
    public abstract Vector subInPlace(double scalar);

    /**
     * Subtracts another vector from the current vector and returns the result in a new vector.
     *
     * @param   vector  The vector to subtract from the current vector.
     * @return          A new vector holding the result of the subtraction.
     */
    public abstract Vector sub(Vector vector);

    /**
     * Subtracts another vector from the current vector and replaces the current vector with the result.
     *
     * @param   vector  The vector to subtract from the current vector.
     */
    public abstract Vector subInPlace(Vector vector);

    /**
     * Multiplies another vector with the current vector element-wise and returns the result in a new vector.
     *
     * @param   vector  The vector to multiply with the current vector element-wise.
     * @return          A new vector holding the result of the multiplication.
     */
    public abstract Vector multElementwise(Vector vector);

    /**
     * Multiplies another vector with the current vector element-wise and replaces the current vector with the result.
     *
     * @param   vector  The vector to multiply with the current vector element-wise.
     */
    public abstract Vector multElementwiseInPlace(Vector vector);

    /**
     * Divides another vector with the current vector element-wise and returns the result in a new vector.
     *
     * @param   vector  The vector to divide with the current vector element-wise.
     * @return          A new vector holding the result of the division.
     */
    public abstract Vector divElementwise(Vector vector);

    /**
     * Divides another vector with the current vector element-wise and replaces the current vector with the result.
     *
     * @param   vector  The vector to divide with the current vector element-wise.
     */
    public abstract Vector divElementwiseInPlace(Vector vector);

    /**
     * Multiplies the current vector with a scalar and returns the result in a new vector.
     *
     * @param   scalar  The scalar with which to multiply the current vector.
     * @return          A new vector holding the result of the multiplication.
     */
    public abstract Vector mult(double scalar);

    /**
     * Multiplies the current vector with a scalar and replaces the current vector with the result.
     *
     * @param   scalar  The scalar with which to multiply the current vector.
     */
    public abstract Vector multInPlace(double scalar);

    /**
     * Divides the current vector with a scalar and returns the result in a new vector.
     *
     * @param   scalar  The scalar with which to divide the current vector.
     * @return          A new vector holding the result of the division.
     */
    public abstract Vector div(double scalar);

    /**
     * Divides the current vector with a scalar and replaces the current vector with the result.
     *
     * @param   scalar  The scalar with which to divide the current vector.
     */
    public abstract Vector divInPlace(double scalar);

    /**
     * Performs the saxpy operation, as it is named in LAPACK. Let us denote the current vector by \(\boldsymbol{y}\).
     * Given a scalar \(\alpha\) and another vector \(\boldsymbol{x}\), this function returns the value of
     * \(\boldsymbol{y}+\alpha\boldsymbol{x}\).
     *
     * @param   scalar  The scalar \(\alpha\).
     * @param   vector  The vector \(\boldsymbol{x}\).
     * @return          The value of \(\boldsymbol{y}+\alpha\boldsymbol{x}\).
     */
    public abstract Vector saxpy(double scalar, Vector vector);

    /**
     * Performs the saxpy operation, as it is named in LAPACK, in-place. Let us denote the current vector by
     * \(\boldsymbol{y}\). Given a scalar \(\alpha\) and another vector \(\boldsymbol{x}\), this function replaces
     * this vector with the value of \(\boldsymbol{y}+\alpha\boldsymbol{x}\) and returns it.
     *
     * @param   scalar  The scalar \(\alpha\).
     * @param   vector  The vector \(\boldsymbol{x}\).
     * @return          The value of \(\boldsymbol{y}+\alpha\boldsymbol{x}\).
     */
    public abstract Vector saxpyInPlace(double scalar, Vector vector);

    /**
     * Computes the inner product (also known as the dot product) between the current vector and another vector.
     *
     * @param   vector  The vector used to compute the inner product with the current vector.
     * @return          The resulting inner product value.
     */
    public final double dot(Vector vector) {
        return inner(vector);
    }

    /**
     * Computes the inner product (also known as the dot product) between the current vector and another vector.
     *
     * @param   vector  The vector used to compute the inner product with the current vector.
     * @return          The resulting inner product value.
     */
    public abstract double inner(Vector vector);

    /**
     * Computes the outer product between the current vector and another vector and returns the result in a new matrix.
     *
     * @param   vector  The vector used to compute the outer product with the current vector.
     * @return          A new matrix containing the result of the outer product operation.
     */
    public abstract Matrix outer(Vector vector);

    /**
     * Performs the gaxpy operation, as it is named in LAPACK. Let us denote the current vector by \(\boldsymbol{y}\).
     * Given a matrix \(\A\) and another vector \(\boldsymbol{x}\), this function returns the value of
     * \(\boldsymbol{y}+A\boldsymbol{x}\).
     *
     * @param   matrix  The matrix \(\A\).
     * @param   vector  The vector \(\boldsymbol{x}\).
     * @return          The value of \(\boldsymbol{y}+A\boldsymbol{x}\).
     *
     * @throws  java.lang.IllegalArgumentException  The row dimension of the matrix must agree with the size of the
     *                                              current vector and the column dimension of the matrix must agree
     *                                              with the size of the provided vector.
     */
    public abstract Vector gaxpy(Matrix matrix, Vector vector);

    /**
     * Performs the gaxpy operation, as it is named in LAPACK, in-place. Let us denote the current vector by
     * \(\boldsymbol{y}\). Given a matrix \(A\) and another vector \(\boldsymbol{x}\), this function replaces
     * this vector with the value of \(\boldsymbol{y}+A\boldsymbol{x}\) and returns it.
     *
     * @param   matrix  The matrix \(A\).
     * @param   vector  The vector \(\boldsymbol{x}\).
     * @return          The value of \(\boldsymbol{y}+A\boldsymbol{x}\).
     *
     * @throws  java.lang.IllegalArgumentException  The row dimension of the matrix must agree with the size of the
     *                                              current vector and the column dimension of the matrix must agree
     *                                              with the size of the provided vector.
     */
    public abstract Vector gaxpyInPlace(Matrix matrix, Vector vector);

    /**
     * Let us denote the current vector by \(\boldsymbol{y}\). Given a matrix \(\A\), this function returns the value of
     * \(\boldsymbol{y}^{\top}A\).
     *
     * @param   matrix  The matrix \(A\).
     * @return          The value of \(\boldsymbol{y}^{\top}A\).
     *
     * @throws  java.lang.IllegalArgumentException  The row dimension of the matrix must agree with the size of the
     *                                              vector.
     */
    public abstract Vector transMult(Matrix matrix);

    /**
     * Checks whether the provided vector has the same size as this vector. If the sizes of the two vectors do not agree
     * an exception is thrown.
     *
     * @param   vector  The vector whose size to check.
     *
     * @throws  IllegalArgumentException    Vector sizes must agree.
     */
    protected void checkVectorSize(Vector vector) {
        if (vector.size() != this.size()) {
            throw new IllegalArgumentException("Vector sizes must agree.");
        }
    }

    /**
     * Writes the contents of this vector to the provided output stream.
     *
     * @param   outputStream    The output stream to write the contents of this vector to.
     * @throws  IOException
     */
    public abstract void writeToStream(ObjectOutputStream outputStream) throws IOException;

    /** {@inheritDoc} */
    @Override
    public abstract boolean equals(Object obj);

    /** {@inheritDoc} */
    @Override
    public abstract int hashCode();
}
