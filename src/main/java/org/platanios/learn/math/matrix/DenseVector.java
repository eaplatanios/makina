package org.platanios.learn.math.matrix;

import java.util.function.Function;

/**
 * Implements a class representing dense vectors and supporting operations related to them. The dense vectors are stored
 * in an internal one-dimensional array.
 *
 * @author Emmanouil Antonios Platanios
 */
public class DenseVector extends Vector {
    /** The size of the vector. */
    private final int size;

    /** Array for internal storage of the vector elements. */
    private double[] array;

    /**
     * Constructs a vector with the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     */
    public DenseVector(int size) {
        this.size = size;
        array = new double[size];
    }

    /**
     * Constructs a vector with the given size and fills it with the provided value.
     *
     * @param   size    The size of the vector.
     * @param   value   The value with which to fill the vector.
     */
    public DenseVector(int size, double value) {
        this.size = size;
        array = new double[size];
        for (int i = 0; i < size; i++) {
            array[i] = value;
        }
    }

    /**
     * Constructs a vector from a one-dimensional array.
     *
     * @param   array   One-dimensional array of doubles.
     */
    public DenseVector(double[] array) {
        size = array.length;
        this.array = array;
    }

    /** {@inheritDoc} */
    @Override
    public VectorType type() {
        return VectorType.DENSE;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector copy() {
        DenseVector resultVector = new DenseVector(size);
        double[] resultVectorArray = resultVector.getArray();
        System.arraycopy(array, 0, resultVectorArray, 0, size);
        return resultVector;
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
    public double get(int index) {
        try {
            return array[index];
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("The provided index is out of bounds.");
        }
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector get(int initialIndex, int finalIndex) {
        DenseVector resultVector = new DenseVector(finalIndex - initialIndex + 1);
        double[] resultVectorArray = resultVector.getArray();
        try {
            System.arraycopy(array, initialIndex, resultVectorArray, 0, finalIndex + 1 - initialIndex);
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided sub-vector indexes are out of bounds."
            );
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public DenseVector get(int[] indexes) {
        DenseVector resultVector = new DenseVector(indexes.length);
        double[] resultVectorArray = resultVector.getArray();
        try {
            for (int i = 0; i < indexes.length; i++) {
                resultVectorArray[i] = array[indexes[i]];
            }
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided sub-vector indexes are out of bounds."
            );
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public void set(int index, double value) {
        try {
            array[index] = value;
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException("The provided index is out of bounds.");
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(int initialIndex, int finalIndex, Vector vector) {
        try {
            for (int i = initialIndex; i <= finalIndex; i++) {
                array[i] = vector.get(i - initialIndex);
            }
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided vector indexes are out of bounds."
            );
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(int[] indexes, Vector vector) {
        try {
            for (int i = 0; i < indexes.length; i++) {
                array[indexes[i]] = vector.get(i);
            }
        } catch(ArrayIndexOutOfBoundsException e) {
            throw new ArrayIndexOutOfBoundsException(
                    "Some or all of the provided vector indexes are out of bounds."
            );
        }
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
    public double min() {
        double minValue = array[0];
        for (int i = 1; i < size; i++) {
            minValue = Math.min(minValue, array[i]);
        }
        return minValue;
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
    public DenseVector addInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            array[i] += scalar;
        }
        return this;
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
    public DenseVector subInPlace(double scalar) {
        for (int i = 0; i < size; i++) {
            array[i] -= scalar;
        }
        return this;
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
        for (int i = 0; i < size; i++) {
            array[i] += scalar * vector.get(i);
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public double inner(Vector vector) {
        checkVectorSize(vector);
        double dotProduct = 0;
        for (int i = 0; i < size; i++) {
            dotProduct += array[i] * vector.get(i);
        }
        return dotProduct;
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
     * Checks whether the provided vector has the same size as this vector. If the sizes of the two vectors do not agree
     * an exception is thrown.
     *
     * @param   vector  The vector whose size to check.
     *
     * @throws  IllegalArgumentException    Vector sizes must agree.
     */
    private void checkVectorSize(Vector vector) {
        if (vector.size() != size) {
            throw new IllegalArgumentException("Vector sizes must agree.");
        }
    }
}
