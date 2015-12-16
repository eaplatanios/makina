package org.platanios.learn.math.matrix;

import org.platanios.learn.math.MathUtilities;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidObjectException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.StringJoiner;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Implements a class representing dense vectors and supporting operations related to them. The dense vectors are stored
 * in an internal one-dimensional array.
 * TODO: Add toSparseVector() method (or appropriate constructors).
 * TODO: Add Builder class and remove constructors.
 *
 * @author Emmanouil Antonios Platanios
 */
public class DenseVector extends Vector {
    /** The size of the vector. */
    protected int size;

    /** Array for internal storage of the vector elements. */
    protected double[] array;

    /**
     * Constructs a dense vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     */
    protected DenseVector(int size) {
        this.size = size;
        array = new double[size];
    }

    /**
     * Constructs a dense vector of the given size and fills it with the provided value.
     *
     * @param   size    The size of the vector.
     * @param   value   The value with which to fill the vector.
     */
    protected DenseVector(int size, double value) {
        this.size = size;
        array = new double[size];
        for (int i = 0; i < size; i++) {
            array[i] = value;
        }
    }

    /**
     * Constructs a dense vector from a one-dimensional array.
     *
     * @param   elements    One-dimensional array of values with which to fill the vector.
     */
    protected DenseVector(double[] elements) {
        size = elements.length;
        array = Arrays.copyOf(elements, size);
    }

    /** {@inheritDoc} */
    @Override
    public VectorType type() {
        return VectorType.DENSE;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector copy() {
        return new DenseVector(array);
    }

    /** {@inheritDoc} */
    @Override
    public double[] getDenseArray() {
        double[] resultArray = new double[size];
        System.arraycopy(array, 0, resultArray, 0, size);
        return resultArray;
    }

    /** {@inheritDoc} */
    @Override
    public int size() {
        return size;
    }

    /** {@inheritDoc} */
    @Override
    public int cardinality() {
        int numberOfNonzeroElements = 0;
        for (double element : array)
            if (element <= epsilon)
                numberOfNonzeroElements++;
        return numberOfNonzeroElements;
    }

    /** {@inheritDoc} */
    @Override
    public double get(int index) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        return array[index];
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector get(int initialIndex, int finalIndex) {
        if (initialIndex < 0 || initialIndex >= size || finalIndex < 0 || finalIndex >= size) {
            throw new IllegalArgumentException(
                    "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        if (initialIndex > finalIndex) {
            throw new IllegalArgumentException("The initial index must be smaller or equal to the final index.");
        }
        DenseVector resultVector = new DenseVector(finalIndex - initialIndex + 1);
        double[] resultVectorArray = resultVector.getArray();
        System.arraycopy(array, initialIndex, resultVectorArray, 0, finalIndex + 1 - initialIndex);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector get(int... indexes) {
        DenseVector resultVector = new DenseVector(indexes.length);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < indexes.length; i++) {
            if (i < 0 || i >= size) {
                throw new IllegalArgumentException(
                        "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
                );
            }
            resultVectorArray[i] = array[indexes[i]];
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public void set(int index, double value) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        array[index] = value;
    }

    /** {@inheritDoc} */
    @Override
    public void set(int initialIndex, int finalIndex, Vector vector) {
        if (initialIndex < 0 || initialIndex >= size || finalIndex < 0 || finalIndex >= size) {
            throw new IllegalArgumentException(
                    "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        if (initialIndex > finalIndex) {
            throw new IllegalArgumentException("The initial index must be smaller or equal to the final index");
        }
        for (int i = initialIndex; i <= finalIndex; i++) {
            array[i] = vector.get(i - initialIndex);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(int[] indexes, Vector vector) {
        for (int i = 0; i < indexes.length; i++) {
            if (indexes[i] < 0 || indexes[i] >= size) {
                throw new IllegalArgumentException(
                        "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
                );
            }
            array[indexes[i]] = vector.get(i);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++)
            array[i] = vector.get(i);
    }

    /** {@inheritDoc} */
    @Override
    public void setAll(double value) {
        for (int i = 0; i < size; i++) {
            array[i] = value;
        }
    }

    /** {@inheritDoc} */
    @Override
    public double max() {
        double maxValue = array[0];
        for (int i = 1; i < size; i++) {
            maxValue = Math.max(maxValue, array[i]);
        }
        return maxValue;
    }

    /** {@inheritDoc} */
    @Override
    public Vector maxElementwise(double value) {
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++)
            resultVectorArray[i] = Math.max(array[i], value);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector maxElementwiseInPlace(double value) {
        for (int i = 0; i < size; i++)
            array[i] = Math.max(array[i], value);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector maxElementwise(Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++)
            resultVectorArray[i] = Math.max(array[i], vector.get(i));
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector maxElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++)
            array[i] = Math.max(array[i], vector.get(i));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public double min() {
        double minValue = array[0];
        for (int i = 1; i < size; i++) {
            minValue = Math.min(minValue, array[i]);
        }
        return minValue;
    }

    /** {@inheritDoc} */
    @Override
    public Vector minElementwise(double value) {
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++)
            resultVectorArray[i] = Math.min(array[i], value);
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector minElementwiseInPlace(double value) {
        for (int i = 0; i < size; i++)
            array[i] = Math.min(array[i], value);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Vector minElementwise(Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++)
            resultVectorArray[i] = Math.min(array[i], vector.get(i));
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public Vector minElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++)
            array[i] = Math.min(array[i], vector.get(i));
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public double sum() {
        double sum = array[0];
        for (int i = 1; i < size; i++) {
            sum += array[i];
        }
        return sum;
    }

    /** {@inheritDoc} */
    @Override
    public double norm(VectorNorm normType) {
        return normType.compute(array);
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector add(double scalar) {
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] + scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector addInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            array[i] += scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector add(Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] + vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector addInPlace(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] += vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector sub(double scalar) {
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] - scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector subInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            array[i] -= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector sub(Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] - vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector subInPlace(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] -= vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector multElementwise(Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] * vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector multElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] *= vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector divElementwise(Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] / vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector divElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] /= vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mult(double scalar) {
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] * scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector multInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            array[i] *= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector div(double scalar) {
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] / scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector divInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            array[i] /= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector saxpy(double scalar, Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = array[i] + scalar * vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector saxpyInPlace(double scalar, Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] += scalar * vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector saxpyPlusConstant(double scalar, Vector vector) {
        if (vector.size() + 1 != this.size())
            throw new IllegalArgumentException("The provided vector size must be 1 less than the current vector size.");
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size - 1; i++)
            resultVectorArray[i] = array[i] + scalar * vector.get(i);
        resultVectorArray[size - 1] = array[size - 1] + scalar;
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector saxpyPlusConstantInPlace(double scalar, Vector vector) {
        if (vector.size() + 1 != this.size())
            throw new IllegalArgumentException("The provided vector size must be 1 less than the current vector size.");
        for (int i = 0; i < size - 1; i++)
            array[i] += scalar * vector.get(i);
        array[size - 1] += scalar;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public double inner(Vector vector) {
        checkVectorSize(vector);
        double dotProduct = 0;
        for (int i = 0; i < size; i++)
            dotProduct += array[i] * vector.get(i);
        return dotProduct;
    }

    /** {@inheritDoc} */
    @Override
    public double innerPlusConstant(Vector vector) {
        if (vector.size() + 1 != this.size())
            throw new IllegalArgumentException("The provided vector size must be 1 less than the current vector size.");
        double dotProduct = 0;
        for (int i = 0; i < size - 1; i++)
            dotProduct += array[i] * vector.get(i);
        return dotProduct + array[size - 1];
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector hypotenuse(Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = MathUtilities.computeHypotenuse(array[i], vector.get(i));
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector hypotenuseInPlace(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] = MathUtilities.computeHypotenuse(array[i], vector.get(i));
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector hypotenuseFast(Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = Math.sqrt(array[i] * array[i] + vector.get(i) * vector.get(i));
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector hypotenuseFastInPlace(Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] = Math.sqrt(array[i] * array[i] + vector.get(i) * vector.get(i));
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector map(Function<Double, Double> function) {
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = function.apply(array[i]);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapInPlace(Function<Double, Double> function) {
        for (int i = 0; i < size; i++) {
            array[i] = function.apply(array[i]);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapBiFunction(BiFunction<Double, Double, Double> function, Vector vector) { // TODO: Check other vector type.
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = function.apply(array[i], vector.get(i));
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapBiFunctionInPlace(BiFunction<Double, Double, Double> function, Vector vector) { // TODO: Check other vector type.
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] = function.apply(array[i], vector.get(i));
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapAdd(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = function.apply(array[i]) + vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapAddInPlace(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] = function.apply(array[i]) + vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapSub(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = function.apply(array[i]) - vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapSubInPlace(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] = function.apply(array[i]) - vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapMultElementwise(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = function.apply(array[i]) * vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapMultElementwiseInPlace(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] = function.apply(array[i]) * vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapDivElementwise(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < size; i++) {
            resultVectorArray[i] = function.apply(array[i]) / vector.get(i);
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector mapDivElementwiseInPlace(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        for (int i = 0; i < size; i++) {
            array[i] = function.apply(array[i]) / vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Matrix outer(Vector vector) {
        checkVectorSize(vector);
        Matrix resultMatrix = new Matrix(size, size);
        double[][] resultMatrixArray = resultMatrix.getArray();
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                resultMatrixArray[i][j] = array[i] * vector.get(j);
            }
        }
        return resultMatrix;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector gaxpy(Matrix matrix, Vector vector) {
        if (matrix.getRowDimension() != size) {
            throw new IllegalArgumentException(
                    "The row dimension of the matrix must agree with the size of the current vector."
            );
        }
        if (matrix.getColumnDimension() != vector.size()) {
            throw new IllegalArgumentException(
                    "The column dimension of the matrix must agree with the size of the provided vector."
            );
        }
        DenseVector resultVector = new DenseVector(matrix.getColumnDimension());
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            resultVectorArray[i] = array[i];
            for (int j = 0; j < vector.size(); j++) {
                resultVectorArray[i] += vector.get(j) * matrix.getElement(i, j);
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector gaxpyInPlace(Matrix matrix, Vector vector) {
        if (matrix.getRowDimension() != size) {
            throw new IllegalArgumentException(
                    "The row dimension of the matrix must agree with the size of the current vector."
            );
        }
        if (matrix.getColumnDimension() != vector.size()) {
            throw new IllegalArgumentException(
                    "The column dimension of the matrix must agree with the size of the provided vector."
            );
        }
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < vector.size(); j++) {
                array[i] += vector.get(j) * matrix.getElement(i, j);
            }
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector transMult(Matrix matrix) {
        if (matrix.getRowDimension() != size) {
            throw new IllegalArgumentException(
                    "The row dimension of the matrix must agree with the size of the vector."
            );
        }
        DenseVector resultVector = new DenseVector(matrix.getColumnDimension());
        double[] resultVectorArray = resultVector.getArray();
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            for (int j = 0; j < size; j++) {
                resultVectorArray[i] += array[j] * matrix.getElement(i, j);
            }
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector prepend(double value) {
        size += 1;
        double[] temporaryArray = array;
        array = new double[size];
        System.arraycopy(temporaryArray, 0, array, 1, size - 1);
        array[0] = value;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector append(double value) {
        size += 1;
        array = Arrays.copyOf(array, size);
        array[size - 1] = value;
        return this;
    }

    /**
     * Constructs and returns a vector with the provided size, filled with ones.
     *
     * @param   size    The size of the vector.
     * @return          A vector with the provided size, filled with ones.
     */
    public static DenseVector generateOnesVector(int size) {
        DenseVector onesVector = new DenseVector(size);
        double[] onesVectorArray = onesVector.getArray();
        for (int i = 0; i < size; i++) {
            onesVectorArray[i] = 1.0;
        }
        return onesVector;
    }

    /**
     * Constructs and returns a vector with the provided size, filled with random values ranging from {@code 0.0} to 
     * {@code 1.0}.
     *
     * @param   size    The size of the random vector.
     * @return          A vector with the provided size, filled with random values ranging from {@code 0.0} to
     *                  {@code 1.0}.
     */
    public static DenseVector generateRandomVector(int size) {
        DenseVector randomVector = new DenseVector(size);
        double[] randomVectorArray = randomVector.getArray();
        for (int i = 0; i < size; i++) {
            randomVectorArray[i] = Math.random();
        }
        return randomVector;
    }

    /**
     * Gets a pointer to the internal one-dimensional array.
     *
     * @return  A pointer to the internal one-dimensional array.
     */
    private double[] getArray() {
        return array;
    }

    /**
     * Returns an iterator used for iterating over all elements of this vector.
     *
     * @return  An iterator used for iterating over all elements of this vector.
     */
    @Override
    public Iterator<VectorElement> iterator() {
        return new DenseVectorIterator();
    }

    /**
     * An iterator class used for iterating over {@link DenseVector} elements.
     */
    private class DenseVectorIterator implements Iterator<Vector.VectorElement> {
        /** Current element index in the vector. */
        private int nextElementIndex = 0;

        /** {@inheritDoc} */
        @Override
        public boolean hasNext() {
            return nextElementIndex < DenseVector.this.size;
        }

        /** {@inheritDoc} */
        @Override
        public Vector.VectorElement next() {
            if (!hasNext())
                throw new NoSuchElementException();

            Vector.VectorElement vectorElement = new Vector.VectorElement(nextElementIndex,
                                                                          DenseVector.this.array[nextElementIndex]);
            nextElementIndex++;

            return vectorElement;
        }

        /** {@inheritDoc} */
        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object object) {
        if (!(object instanceof DenseVector))
            return false;
        if (object == this)
            return true;

        DenseVector that = (DenseVector) object;

        if (size != that.size)
            return false;
        for (int index = 0; index < size; index++)
            if ((Math.abs(array[index] - that.array[index]) >= epsilon))
                return false;

        return true;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringJoiner stringJoiner = new StringJoiner(",", "[", "]");
        for (double element : array)
            stringJoiner.add(String.valueOf(element));
        return stringJoiner.toString();
    }

    /** {@inheritDoc} */
    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException {
        if (includeType)
            UnsafeSerializationUtilities.writeInt(outputStream, type().ordinal());
        UnsafeSerializationUtilities.writeInt(outputStream, size);
        UnsafeSerializationUtilities.writeDoubleArray(outputStream, array);
    }

    /**
     * Deserializes the dense vector stored in the provided input stream and returns it.
     *
     * @param   inputStream Input stream from which the dense vector will be "read".
     * @return              The dense vector obtained from the provided input stream.
     * @throws  IOException
     */
    public static DenseVector read(InputStream inputStream, boolean includeType) throws IOException {
        if (includeType) {
            VectorType vectorType = VectorType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
            if (vectorType != VectorType.DENSE)
                throw new InvalidObjectException("The stored vector is of type " + vectorType.name() + "!");
        }
        int size = UnsafeSerializationUtilities.readInt(inputStream);
        DenseVector vector = new DenseVector(size);
        vector.array = UnsafeSerializationUtilities.readDoubleArray(inputStream, size, 4096);
        return vector;
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getEncoder(boolean includeType) {
        return new Encoder(includeType);
    }

    /**
     * Encoder class for dense vectors. This class extends the Java {@link InputStream} class and can be used to copy
     * dense vector instances into other locations (e.g., in a database). Note that this encoder uses the underlying
     * vector and so, if that vector is changed, the output of this encoder might be changed and even become corrupt.
     *
     * The dense vector is serialized in the following way: (i) the size of the vector is encoded first, and (ii) the
     * elements of the underlying array are encoded next, in the order in which they appear in the array.
     */
    protected class Encoder extends InputStream {
        /** A pointer in memory used to represent the current position while serializing different fields of the object
         * that is being serialized. */
        long position;
        /** The largest value that the {@link #position} pointer can take (i.e., this pointer represents the end of the
         * field that is currently being serialized, in memory). */
        long endPosition;
        /** The current state of the encoder, representing which field of the object is currently being serialized. */
        EncoderState state;

        /** The memory address offset of {@link #size} from the base address of the dense vector instance that is being
         * encoded. */
        final long sizeFieldOffset;
        /** The {@link VectorType} ordinal number of the type of the vector being encoded
         * (i.e., {@link VectorType#DENSE}). */
        final int type;
        /** Boolean value indicating whether or not to also encode the type of the current vector
         * (i.e., {@link VectorType#DENSE}). */
        final boolean includeType;

        /** Constructs an encoder object from the current vector. */
        public Encoder(boolean includeType) {
            long typeFieldOffset;
            try {
                sizeFieldOffset = UNSAFE.objectFieldOffset(DenseVector.class.getDeclaredField("size"));
                typeFieldOffset = UNSAFE.objectFieldOffset(Encoder.class.getDeclaredField("type"));
            } catch (NoSuchFieldException e) {
                throw new RuntimeException(e);
            }
            if (!includeType) {
                position = sizeFieldOffset;
                endPosition = sizeFieldOffset + 4;
                state = EncoderState.SIZE;
            } else {
                position = typeFieldOffset;
                endPosition = typeFieldOffset + 4;
                state = EncoderState.TYPE;
            }
            type = VectorType.DENSE.ordinal();
            this.includeType = includeType;
        }

        /** {@inheritDoc} */
        @Override
        public int read() {
            switch(state) {
                case TYPE:
                    if (position == endPosition) {
                        position = sizeFieldOffset;
                        endPosition = sizeFieldOffset + 4;
                        state = EncoderState.SIZE;
                    } else {
                        return UNSAFE.getByte(this, position++);
                    }
                case SIZE:
                    if (position == endPosition) {
                        position = DOUBLE_ARRAY_OFFSET;
                        endPosition = DOUBLE_ARRAY_OFFSET + (size << 3);
                        state = EncoderState.ARRAY;
                    } else {
                        return UNSAFE.getByte(DenseVector.this, position++);
                    }
                case ARRAY:
                    if (position == endPosition)
                        return -1;
                    else
                        return UNSAFE.getByte(array, position++);
            }
            return -1;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] destination) {
            return read(destination, 0, destination.length);
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte destination[], int offset, int length) {
            if (destination == null)
                throw new NullPointerException();
            if (offset < 0 || length < 0 || length > destination.length - offset)
                throw new IndexOutOfBoundsException();
            int bytesRead;
            switch(state) {
                case TYPE:
                    bytesRead = readBytes(this, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = sizeFieldOffset;
                        endPosition = sizeFieldOffset + 4;
                        state = EncoderState.SIZE;
                    }
                case SIZE:
                    bytesRead = readBytes(DenseVector.this, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = DOUBLE_ARRAY_OFFSET;
                        endPosition = DOUBLE_ARRAY_OFFSET + (size << 3);
                        state = EncoderState.ARRAY;
                    }
                case ARRAY:
                    return readBytes(array, destination, offset, length);
            }
            return -1;
        }

        /**
         * Reads up to {@code length} bytes of data from the input stream into an array of bytes and returns the number
         * of bytes read.
         *
         * @param   source      The source object to read data from.
         * @param   destination The byte array to write data into.
         * @param   offset      The memory address offset into the destination byte array, to start writing data from.
         * @param   length      The maximum number of bytes to read from {@code source} and write to
         *                      {@code destination}.
         * @return              The number of bytes read from {@code source} and written to {@code destination}.
         */
        private int readBytes(Object source, byte[] destination, int offset, int length) {
            long numberOfBytesToRead = Math.min(endPosition - position, length);
            if (numberOfBytesToRead > 0) {
                UNSAFE.copyMemory(source,
                                  position,
                                  destination,
                                  BYTE_ARRAY_OFFSET + offset,
                                  numberOfBytesToRead);
                position += numberOfBytesToRead;
                return (int) numberOfBytesToRead;
            } else {
                return -1;
            }
        }
    }

    /** Enumeration containing the possible encoder states used within the {@link Encoder} class. */
    private enum EncoderState {
        /** Represents the state the encoder is in, while encoding the type of the dense vector. */
        TYPE,
        /** Represents the state the encoder is in, while encoding the size of the dense vector. */
        SIZE,
        /** Represents the state the encoder is in, while encoding the underlying array of the dense vector. */
        ARRAY
    }
}
