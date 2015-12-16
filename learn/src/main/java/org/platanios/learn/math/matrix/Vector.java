package org.platanios.learn.math.matrix;

import org.platanios.learn.math.MathUtilities;
import sun.misc.Unsafe;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.util.Iterator;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Abstract class that needs to be extended by classes representing vectors and supporting operations related to
 * vectors.
 *
 * TODO: Force subclasses to implement the hashCode() method.
 * TODO: Allow transposing all matrix arguments.
 * TODO: Add vector iterators support (over all elements and nonzero elements only).
 *
 * @author Emmanouil Antonios Platanios
 */
public abstract class Vector implements Iterable<Vector.VectorElement> {
    /** Obtain a singleton instance of the {@link sun.misc.Unsafe} class for use within the unsafe serialization
     * mechanism used for all the {@link Vector} subclasses. */
    protected static final Unsafe UNSAFE;
    static
    {
        try
        {
            Field field = Unsafe.class.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            UNSAFE = (Unsafe)field.get(null);
        }
        catch (Exception e)
        {
            throw new RuntimeException(e);
        }
    }
    /** The offset, in bytes, between the base memory address of a byte array and its first element. */
    protected static final long BYTE_ARRAY_OFFSET = UNSAFE.arrayBaseOffset(byte[].class);
    /** The offset, in bytes, between the base memory address of a integer array and its first element. */
    protected static final long INT_ARRAY_OFFSET = UNSAFE.arrayBaseOffset(int[].class);
    /** The offset, in bytes, between the base memory address of a double array and its first element. */
    protected static final long DOUBLE_ARRAY_OFFSET = UNSAFE.arrayBaseOffset(double[].class);

    /** The threshold value for elements to be considered equal to zero when counting the number of non-zero elements of
     * this vector (i.e., in method {@link #cardinality()}) and when handling sparse vectors. */
    protected static final double epsilon = MathUtilities.computeMachineEpsilonDouble();

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
     * Returns a dense vector copy of the current vector.
     *
     * @return  A dense vector copy of this vector.
     */
    public DenseVector toDenseVector() {
        return new DenseVector(getDenseArray());
    }

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
    public abstract Vector get(int... indexes);

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
     * Sets a sub-vector of this vector to the provided vector values.
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
     * Sets the elements of this vector to the provided vector values.
     *
     * @param   vector      The vector to whose values we set the values of this vector.
     *
     * @throws  java.lang.IllegalArgumentException  The provided vector size does not match this vector's size.
     */
    public abstract void set(Vector vector);

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
     * Gets the element-wise maximum value between the elements of the current vector and the provided value.
     *
     * @return  A new vector holding the result of the element-wise maximum operation.
     */
    public abstract Vector maxElementwise(double value);

    /**
     * Sets the elements of the current vector to the element-wise maximum value between the elements of the current
     * vector and the provided value.
     *
     * @return  The current vector holding the result of the element-wise maximum operation.
     */
    public abstract Vector maxElementwiseInPlace(double value);

    /**
     * Gets the element-wise maximum value between the elements of the current vector and the elements of the provided
     * vector.
     *
     * @return  A new vector holding the result of the element-wise maximum operation.
     */
    public abstract Vector maxElementwise(Vector vector);

    /**
     * Sets the elements of the current vector to the element-wise maximum value between the elements of the current
     * vector and the elements of the provided vector.
     *
     * @return  The current vector holding the result of the element-wise maximum operation.
     */
    public abstract Vector maxElementwiseInPlace(Vector vector);

    /**
     * Gets the minimum value of all elements in this vector.
     *
     * @return  The minimum value of all elements in this vector.
     */
    public abstract double min();

    /**
     * Gets the element-wise minimum value between the elements of the current vector and the provided value.
     *
     * @return  A new vector holding the result of the element-wise minimum operation.
     */
    public abstract Vector minElementwise(double value);

    /**
     * Sets the elements of the current vector to the element-wise minimum value between the elements of the current
     * vector and the provided value.
     *
     * @return  The current vector holding the result of the element-wise minimum operation.
     */
    public abstract Vector minElementwiseInPlace(double value);

    /**
     * Gets the element-wise minimum value between the elements of the current vector and the elements of the provided
     * vector.
     *
     * @return  A new vector holding the result of the element-wise minimum operation.
     */
    public abstract Vector minElementwise(Vector vector);

    /**
     * Sets the elements of the current vector to the element-wise minimum value between the elements of the current
     * vector and the elements of the provided vector.
     *
     * @return  The current vector holding the result of the element-wise minimum operation.
     */
    public abstract Vector minElementwiseInPlace(Vector vector);

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
     * @return          The current vector holding the result of the addition.
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
     * @return          The current vector holding the result of the addition.
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
     * @return          The current vector holding the result of the subtraction.
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
     * @return          The current vector holding the result of the subtraction.
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
     * @return          The current vector holding the result of the multiplication.
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
     * @return          The current vector holding the result of the division.
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
     * @return          The current vector holding the result of the multiplication.
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
     * @return          The current vector holding the result of the division.
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
     * Performs a modified saxpy operation, as it is named in LAPACK. Let us denote the current vector by
     * \(\boldsymbol{y}\). Given a scalar \(\alpha\) and another vector \(\boldsymbol{x}\) with size 1 less than the
     * current vector, this function returns the value of \(\boldsymbol{y}+\alpha\boldsymbol{x}\), where for the last
     * element of the current vector, /(\boldsymbol{y}/), we simply add \(\alpha\).
     *
     * @param   scalar  The scalar \(\alpha\).
     * @param   vector  The vector \(\boldsymbol{x}\).
     * @return          The result of this operation.
     */
    public abstract Vector saxpyPlusConstant(double scalar, Vector vector);

    /**
     * Performs a modified saxpy operation, as it is named in LAPACK. Let us denote the current vector by
     * \(\boldsymbol{y}\). Given a scalar \(\alpha\) and another vector \(\boldsymbol{x}\) with size 1 less than the
     * current vector, this function replaces this vector with the value of \(\boldsymbol{y}+\alpha\boldsymbol{x}\),
     * where for the last element of the current vector, /(\boldsymbol{y}/), we simply add \(\alpha\).
     *
     * @param   scalar  The scalar \(\alpha\).
     * @param   vector  The vector \(\boldsymbol{x}\).
     * @return          The result of this operation (i.e., the current vector).
     */
    public abstract Vector saxpyPlusConstantInPlace(double scalar, Vector vector);

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
     * Computes the inner product (also known as the dot product) between the current vector without its last element
     * and another vector, adds to that result the last element of the current vector, and returns the result.
     *
     * @param   vector  The vector used to compute the inner product with the current vector without its last element.
     *                  Note that this vector must have size 1 less than the size of the current vector.
     * @return          The resulting value.
     */
    public final double dotPlusConstant(Vector vector) {
        return innerPlusConstant(vector);
    }

    /**
     * Computes the inner product (also known as the dot product) between the current vector and another vector.
     *
     * @param   vector  The vector used to compute the inner product with the current vector.
     * @return          The resulting inner product value.
     */
    public abstract double inner(Vector vector);

    /**
     * Computes the inner product (also known as the dot product) between the current vector without its last element
     * and another vector, adds to that result the last element of the current vector, and returns the result.
     *
     * @param   vector  The vector used to compute the inner product with the current vector without its last element.
     *                  Note that this vector must have size 1 less than the size of the current vector.
     * @return          The resulting value.
     */
    public abstract double innerPlusConstant(Vector vector);

    /**
     * Computes the square root of the sum of the squares of each pair of vector elements (that is equivalent to
     * computing length of the hypotenuse of a right triangle given the lengths of the other two sides) without having
     * an underflow or an overflow. Denoting the two vectors by \(\boldsymbol{a}\) (the current vector) and
     * \(\boldsymbol{b}\) (the provided vector), respectively, this function computes the quantity:
     * \[c_i=\sqrt{a_i^2+b_i^2},\]
     * where \(\boldsymbol{c}\) is the resulting vector. The result is returned in a new vector. This method is slower
     * than {@link #hypotenuseFast(Vector)}, but it tries to avoid numerical precision errors, while
     * {@link #hypotenuseFast(Vector)} does not.
     *
     * @param   vector  The vector to use as vector \(\boldsymbol{b}\) in the above equation.
     * @return          A new vector holding the result of this operation.
     */
    public abstract Vector hypotenuse(Vector vector);

    /**
     * Computes the square root of the sum of the squares of each pair of vector elements (that is equivalent to
     * computing length of the hypotenuse of a right triangle given the lengths of the other two sides) without having
     * an underflow or an overflow. Denoting the two vectors by \(\boldsymbol{a}\) (the current vector) and
     * \(\boldsymbol{b}\) (the provided vector), respectively, this function computes the quantity:
     * \[c_i=\sqrt{a_i^2+b_i^2},\]
     * where \(\boldsymbol{c}\) is the resulting vector. The current vector is replaced with the result. This method is
     * slower than {@link #hypotenuseFastInPlace(Vector)}, but it tries to avoid numerical precision errors, while
     * {@link #hypotenuseFastInPlace(Vector)} does not.
     *
     * @param   vector  The vector to use as vector \(\boldsymbol{b}\) in the above equation.
     * @return          The current vector holding the result of this operation.
     */
    public abstract Vector hypotenuseInPlace(Vector vector);

    /**
     * Computes the square root of the sum of the squares of each pair of vector elements (that is equivalent to
     * computing length of the hypotenuse of a right triangle given the lengths of the other two sides) without avoiding
     * an underflow or an overflow. Denoting the two vectors by \(\boldsymbol{a}\) (the current vector) and
     * \(\boldsymbol{b}\) (the provided vector), respectively, this function computes the quantity:
     * \[c_i=\sqrt{a_i^2+b_i^2},\]
     * where \(\boldsymbol{c}\) is the resulting vector. The result is returned in a new vector. This method is faster
     * than {@link #hypotenuse(Vector)}, but it does not try to avoid numerical precision errors, while
     * {@link #hypotenuse(Vector)} does try to avoid such errors.
     *
     * @param   vector  The vector to use as vector \(\boldsymbol{b}\) in the above equation.
     * @return          A new vector holding the result of this operation.
     */
    public abstract Vector hypotenuseFast(Vector vector);

    /**
     * Computes the square root of the sum of the squares of each pair of vector elements (that is equivalent to
     * computing length of the hypotenuse of a right triangle given the lengths of the other two sides) without avoiding
     * an underflow or an overflow. Denoting the two vectors by \(\boldsymbol{a}\) (the current vector) and
     * \(\boldsymbol{b}\) (the provided vector), respectively, this function computes the quantity:
     * \[c_i=\sqrt{a_i^2+b_i^2},\]
     * where \(\boldsymbol{c}\) is the resulting vector. The current vector is replaced with the result. This method is
     * faster than {@link #hypotenuseInPlace(Vector)}, but it does not try to avoid numerical precision errors, while
     * {@link #hypotenuseInPlace(Vector)} does try to avoid such errors.
     *
     * @param   vector  The vector to use as vector \(\boldsymbol{b}\) in the above equation.
     * @return          A new vector holding the result of this operation.
     */
    public abstract Vector hypotenuseFastInPlace(Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and returns it in a new
     * vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @return              A new vector holding the result of the operation.
     */
    public abstract Vector map(Function<Double, Double> function);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and replaces the current
     * vector with the result.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @return              The current vector holding the result of the operation.
     */
    public abstract Vector mapInPlace(Function<Double, Double> function);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and the provided vector
     * (the elements of the two vectors with the same index are considered in pairs) and returns it in a new vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to use for second argument of the function.
     * @return              A new vector holding the result of the operation.
     */
    public abstract Vector mapBiFunction(BiFunction<Double, Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and the provided vector
     * (the elements of the two vectors with the same index are considered in pairs) and replaces the current vector
     * with the result.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to use for second argument of the function.
     * @return              The current vector holding the result of the operation.
     */
    public abstract Vector mapBiFunctionInPlace(BiFunction<Double, Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and adding the provided
     * vector to the result, and returns it in a new vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to add to the function result.
     * @return              A new vector holding the result of the operation.
     */
    public abstract Vector mapAdd(Function<Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and adding the provided
     * vector to the result, and replaces the current vector with the result.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to add to the function result.
     * @return              The current vector holding the result of the operation.
     */
    public abstract Vector mapAddInPlace(Function<Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and subtracting the
     * provided vector from the result, and returns it in a new vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to subtract from the function result.
     * @return              A new vector holding the result of the operation.
     */
    public abstract Vector mapSub(Function<Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and subtracting the
     * provided vector from the result, and replaces the current vector with the result.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to subtract from the function result.
     * @return              The current vector holding the result of the operation.
     */
    public abstract Vector mapSubInPlace(Function<Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and multiplying the
     * provided vector with the result element-wise, and returns it in a new vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to multiply with the function result element-wise.
     * @return              A new vector holding the result of the operation.
     */
    public abstract Vector mapMultElementwise(Function<Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and multiplying the
     * provided vector with the result element-wise, and replaces the current vector with the result.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to multiply with the function result element-wise.
     * @return              The current vector holding the result of the operation.
     */
    public abstract Vector mapMultElementwiseInPlace(Function<Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and dividing the
     * provided vector with the result element-wise, and returns it in a new vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to divide with the function result element-wise.
     * @return              A new vector holding the result of the operation.
     */
    public abstract Vector mapDivElementwise(Function<Double, Double> function, Vector vector);

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and dividing the
     * provided vector with the result element-wise, and replaces the current vector with the result.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to divide with the function result element-wise.
     * @return              The current vector holding the result of the operation.
     */
    public abstract Vector mapDivElementwiseInPlace(Function<Double, Double> function, Vector vector);

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
     * @throws  IllegalArgumentException    The row dimension of the matrix must agree with the size of the current
     *                                      vector and the column dimension of the matrix must agree with the size of
     *                                      the provided vector.
     */
    public abstract Vector gaxpy(Matrix matrix, Vector vector);

    /**
     * Performs the gaxpy operation, as it is named in LAPACK, in-place. Let us denote the current vector by
     * \(\boldsymbol{y}\). Given a matrix \(A\) and another vector \(\boldsymbol{x}\), this function replaces this
     * vector with the value of \(\boldsymbol{y}+A\boldsymbol{x}\) and returns it.
     *
     * @param   matrix  The matrix \(A\).
     * @param   vector  The vector \(\boldsymbol{x}\).
     * @return          The value of \(\boldsymbol{y}+A\boldsymbol{x}\).
     *
     * @throws  IllegalArgumentException    The row dimension of the matrix must agree with the size of the current
     *                                      vector and the column dimension of the matrix must agree with the size of
     *                                      the provided vector.
     */
    public abstract Vector gaxpyInPlace(Matrix matrix, Vector vector);

    /**
     * Let us denote the current vector by \(\boldsymbol{y}\). Given a matrix \(\A\), this function returns the value of
     * \(\boldsymbol{y}^{\top}A\).
     *
     * @param   matrix  The matrix \(A\).
     * @return          The value of \(\boldsymbol{y}^{\top}A\).
     *
     * @throws  IllegalArgumentException    The row dimension of the matrix must agree with the size of the vector.
     */
    public abstract Vector transMult(Matrix matrix);

    /**
     * Adds a value to the beginning of this vector, increasing its size by 1.
     *
     * @param   value   The value to prepend to this vector.
     * @return          The current vector after prepending to it the provided value.
     */
    public abstract Vector prepend(double value);

    /**
     * Adds a value to the end of this vector, increasing its size by 1.
     *
     * @param   value   The value to append to this vector.
     * @return          The current vector after appending to it the provided value.
     */
    public abstract Vector append(double value);

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
     * @param   includeType     Boolean value indicating whether the type of the vector is to also be written to the
     *                          output stream.
     * @throws  IOException
     */
    public abstract void write(OutputStream outputStream, boolean includeType) throws IOException;

    /**
     * Returns an encoder (a Java {@link java.io.InputStream} basically). This encoder can be used to copy the current
     * vector somewhere else (e.g., in a database, as a BLOB). Note that if the underlying vector object is modified
     * while some process is reading from the returned stream, then the full object received from the stream might be
     * corrupted (e.g., if it is known that the sum of the current vector elements is one, then that might not be true
     * for the vector read by some other process, if the vector was modified while that process was reading it by using
     * an encoder returned by this method).
     *
     * @param   includeType     Boolean value indicating whether the type of the vector is to also be encoded.
     * @return                  The encoder for the current instance.
     */
    public abstract InputStream getEncoder(boolean includeType);

    /**
     * Returns an iterator used for iterating over a superset of the non-zero elements of the vector. The particular
     * superset used depends on the vector implementation. For example, for {@link DenseVector} that superset is the
     * set of all elements in the vector, whereas for {@link SparseVector} that superset will include all non-zero
     * elements in the vector, but may also include some zero elements.
     *
     * @return  An iterator used for iterating over a superset of the non-zero elements of the vector.
     */
    @Override
    public abstract Iterator<VectorElement> iterator();

    /**
     * Compares the current vector with another object for equality. Note that if the provided object is not a vector
     * object, then this method returns false. Otherwise, it checks for equality of the element values of the two
     * vectors, with some tolerance that is equal to the square root of the smallest possible value that can be
     * represented by a double precision floating point number. A tolerance value is used because double precision
     * floating point number values are compared.
     *
     * Note that the vector objects being compared must also be of the same type (i.e., dense, sparse, etc.). In the
     * case that they are not of the same type a value of false is returned by this method.
     *
     * @param   object  The object with which to compare this vector.
     * @return          True if this vector is equal to the provided object and false if it is not.
     */
    @Override
    public abstract boolean equals(Object object);

    @Override
    public abstract String toString();

    /**
     * Class representing a single vector element. This class contains the index of the element and the value of that
     * element. This class is mainly used by vector iterators.
     */
	public final class VectorElement {
        /** The index of the element in its vector. */
		private final int index;
        /** The value of the element. */
		private final double value;

        /**
         * Construct a single vector element using the provided index and value.
         *
         * @param   index   The index of the element in its vector.
         * @param   value   The value of the element.
         */
		public VectorElement(int index, double value) {
			this.index = index;
			this.value = value;
		}

        /**
         * Returns the index of the element in its vector.
         *
         * @return  The index of the element in its vector.
         */
		public int index() {
			return this.index;
		}

        /**
         * Returns the value of the element.
         *
         * @return  The value of the element.
         */
		public double value() {
			return this.value;
		}
	}
}
