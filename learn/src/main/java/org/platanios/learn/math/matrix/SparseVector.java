package org.platanios.learn.math.matrix;

import org.platanios.learn.math.MathUtilities;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;
import java.io.InvalidObjectException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Implements a class representing sparse vectors and supporting operations related to them. The sparse vector is stored
 * in two internal parallel arrays: one holding the indexes of the nonzero elements of the vector, and one holding the
 * values of the corresponding vector elements. Note that the indexes array is sorted and therefore, changing values of
 * this vector's elements is a slow, O(n), operation, when n is the number of nonzero elements of the vector. Retrieving
 * elements from this vector is an O(log(n)) operation (binary search is used).
 *
 * TODO: Make unsafe versions of all the methods.
 *
 * Performs binary search to retrieve elements within the vector.
 *
 * @author Emmanouil Antonios Platanios
 */
public class SparseVector extends Vector {
    /** The size of the vector. */
    protected int size;

    /** The cardinality of the vector (i.e., the number of nonzero elements of the vector). This value also limits the
     * effective length of {@link #indexes} and {@link #values} (i.e., only the first numberOfNonzeroEntries elements of
     * those two arrays are ever used -- the rest of the values are considered to be equal to zero). */
    protected int numberOfNonzeroEntries;
    /** Integer array for internal storage of the indexes of the non-zero vector elements. This array is always ordered
     * and is "parallel" to the {@link #values} array. */
    protected int[] indexes;
    /** Double array for internal storage of the values of the non-zero vector elements. This array is always "parallel"
     * to the {@link #indexes} array. */
    protected double[] values;

    /**
     * Constructs a sparse vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     */
    public SparseVector(int size) {
        this.size = size;
        numberOfNonzeroEntries = 0;
        indexes = new int[0];
        values = new double[0];
    }

    /**
     * Constructs a sparse vector of the given size and fills it with the values stored in the provided map. The map
     * must contain key-value pairs where the key corresponds to an element index and the value to the corresponding
     * element's value. Note that the map does not need to be an sorted map; all the necessary sorting is performed
     * within this constructor.
     *
     * @param   size            The size of the vector.
     * @param   vectorElements  The map containing the vector indexes and values used to initialize the values of the
     *                          elements of this vector.
     */
    public SparseVector(int size, Map<Integer, Double> vectorElements) {
        this.size = size;
        numberOfNonzeroEntries = vectorElements.size();
        TreeMap<Integer, Double> sortingMap = new TreeMap<>();
        sortingMap.putAll(vectorElements);
        indexes = new int[numberOfNonzeroEntries];
        values = new double[numberOfNonzeroEntries];
        int i = 0;
        for (int key : sortingMap.keySet()) {
            indexes[i] = key;
            values[i] = sortingMap.get(key);
            i++;
        }
    }

    /**
     * Constructs a sparse vector of the given size from the provided parallel arrays containing indexes of vector
     * elements and the values corresponding to those indexes.
     *
     * @param   size    The size of the vector.
     * @param   indexes Integer array containing the indexes of the vector elements for which values are provided. This
     *                  array is "parallel" to the values array, which is also provided as a parameter to this
     *                  constructor.
     * @param   values  Double array containing the values of the vector elements that correspond to the indexes
     *                  provided in the indexes parameter to this constructor. This array is "parallel" to the values
     *                  array, which is also provided as a parameter to this constructor.
     */
    public SparseVector(int size, int[] indexes, double[] values) {
        if (indexes.length != values.length)
            throw new IllegalArgumentException("The indexes array and the values array must have the same length");

        this.size = size;
        numberOfNonzeroEntries = indexes.length;
        this.indexes = Arrays.copyOf(indexes, indexes.length);
        this.values = Arrays.copyOf(values, values.length);
    }

    /**
     * Constructs a sparse vector of the given size from the provided parallel arrays containing indexes of vector
     * elements and the values corresponding to those indexes. Only the first numberOfNonzeroEntries elements of the
     * provided parallel arrays are used and the rest are considered to be equal to zero. This mechanism is used (as
     * opposed to simply resizing the parallel arrays) for time efficiency reasons. Resizing the arrays is slow and the
     * memory cost can generally be considered small.
     *
     * @param   size                    The size of the vector.
     * @param   numberOfNonzeroEntries  The number of elements to consider as corresponding to nonzero vector values in
     *                                  the provided parallel arrays. This number also corresponds to the number of
     *                                  nonzero elements (i.e., the cardinality) of the sparse vector being constructed.
     * @param   indexes                 Integer array containing the indexes of the vector elements for which values are
     *                                  provided. This array is "parallel" to the values array, which is also provided
     *                                  as a parameter to this constructor.
     * @param   values                  Double array containing the values of the vector elements that correspond to the
     *                                  indexes provided in the indexes parameter to this constructor. This array is
     *                                  "parallel" to the values array, which is also provided as a parameter to this
     *                                  constructor.
     */
    public SparseVector(int size, int numberOfNonzeroEntries, int[] indexes, double[] values) {
        if (indexes.length != values.length)
            throw new IllegalArgumentException("The indexes array and the values array must have the same length");

        this.size = size;
        this.numberOfNonzeroEntries = numberOfNonzeroEntries;
        this.indexes = Arrays.copyOf(indexes, indexes.length);
        this.values = Arrays.copyOf(values, values.length);
    }

    /**
     * Constructs a sparse vector from a dense vector. This constructor does not simply transform the dense vector
     * structure into a sparse vector structure, but it also throws away elements of the dense vector that have a value
     * effectively 0 (i.e., absolute value \(<\epsilon\), where \(\epsilon\) is the square root of the smallest possible
     * value that can be represented by a double precision floating point number).
     *
     * @param   vector  The dense vector from which to construct this sparse vector.
     */
    public SparseVector(DenseVector vector) {
        size = vector.size();
        indexes = new int[size];
        values = new double[size];
        int currentIndex = 0;
        for (int i = 0; i < size; i++) {
            if (Math.abs(vector.array[i]) >= epsilon) {
                indexes[currentIndex] = i;
                values[currentIndex] = vector.array[i];
                currentIndex++;
            }
        }
        numberOfNonzeroEntries = currentIndex;
    }

    /**
     * Constructs a sparse vector from another sparse vector. This constructor basically constructs a copy of the
     * provided sparse vector.
     *
     * @param   vector  The sparse vector from which to construct this sparse vector.
     */
    public SparseVector(SparseVector vector) {
        this(vector.size, vector.numberOfNonzeroEntries, vector.indexes, vector.values);
    }

    /**
     * Constructs a sparse vector from another hash vector.
     *
     * @param   vector  The hash vector from which to construct this sparse vector.
     */
    public SparseVector(HashVector vector) {
        throw new UnsupportedOperationException();
    }

    /** {@inheritDoc} */
    @Override
    public VectorType type() {
        return VectorType.SPARSE;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector copy() {
        return new SparseVector(size, indexes, values);
    }

    /** {@inheritDoc} */
    @Override
    public double[] getDenseArray() {
        double[] resultArray = new double[size];
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            resultArray[indexes[i]] = values[i];
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
        return numberOfNonzeroEntries;
    }

    /** {@inheritDoc} */
    @Override
    public double get(int index) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        int valueIndex = Arrays.binarySearch(indexes, index);
        if (valueIndex >= 0) {
            return values[valueIndex];
        } else {
            return 0;
        }
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector get(int initialIndex, int finalIndex) {
        if (initialIndex < 0 || initialIndex >= size || finalIndex < 0 || finalIndex >= size) {
            throw new IllegalArgumentException(
                    "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        if (initialIndex > finalIndex) {
            throw new IllegalArgumentException("The initial index must be smaller or equal to the final index.");
        }
        int startIndex = Arrays.binarySearch(indexes, initialIndex);
        if (startIndex < 0) {
            startIndex = -startIndex - 1;
            if (startIndex < 0)
                startIndex = 0;
        }
        int endIndex = Arrays.binarySearch(indexes, finalIndex);
        if (endIndex < 0) {
            endIndex = -endIndex - 2;
        }
        return new SparseVector(finalIndex - initialIndex + 1,
                                Arrays.copyOfRange(indexes, startIndex, endIndex + 1),
                                Arrays.copyOfRange(values, startIndex, endIndex + 1));
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector get(int[] indexes) {
        Map<Integer, Double> elements = new TreeMap<>();
        for (int i = 0; i < indexes.length; i++) {
            if (i < 0 || i >= size) {
                throw new IllegalArgumentException(
                        "The provided indexes must be between 0 (inclusive) and the size of the vector (exclusive)."
                );
            }
            elements.put(i, get(indexes[i]));
        }
        return new SparseVector(indexes.length, elements);
    }

    /** {@inheritDoc} */
    @Override
    public void set(int index, double value) {
        if (index < 0 || index >= size) {
            throw new IllegalArgumentException(
                    "The provided index must be between 0 (inclusive) and the size of the vector (exclusive)."
            );
        }
        int foundIndex = Arrays.binarySearch(indexes, index);
        if (foundIndex >= 0) {
            values[indexes[foundIndex]] = value;
        } else {
            foundIndex = - foundIndex - 1;
            numberOfNonzeroEntries++;
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            for (int i = 0; i < numberOfNonzeroEntries; i++) {
                if (i < foundIndex) {
                    newIndexes[i] = indexes[i];
                    newValues[i] = values[i];
                } else if (i == foundIndex) {
                    newIndexes[i] = index;
                    newValues[i] = value;
                } else {
                    newIndexes[i] = indexes[i - 1];
                    newValues[i] = values[i - 1];
                }
            }
            indexes = newIndexes;
            values = newValues;
        }
    }

    /** {@inheritDoc} */
    @Override // TODO: It may be possible to make this method faster.
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
            set(i, vector.get(i - initialIndex));
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
            set(indexes[i], vector.get(i));
        }
    }

    /** {@inheritDoc} */
    @Override
    public void setAll(double value) {
        if (Math.abs(value) >= epsilon) {
            numberOfNonzeroEntries = size;
            indexes = new int[numberOfNonzeroEntries];
            values = new double[numberOfNonzeroEntries];
            for (int i = 0; i < numberOfNonzeroEntries; i++) {
                indexes[i] = 0;
                values[i] = value;
            }
        } else {
            numberOfNonzeroEntries = 0;
            indexes = new int[0];
            values = new double[0];
        }
    }

    /** {@inheritDoc} */
    @Override
    public double max() {
        double maxValue = values[0];
        for (int i = 1; i < numberOfNonzeroEntries; i++) {
            maxValue = Math.max(maxValue, values[i]);
        }
        return maxValue;
    }

    /** {@inheritDoc} */
    @Override
    public double min() {
        double minValue = values[0];
        for (int i = 1; i < numberOfNonzeroEntries; i++)
            minValue = Math.min(minValue, values[i]);
        return minValue;
    }

    /** {@inheritDoc} */
    @Override
    public double sum() {
        double sum = values[0];
        for (int i = 1; i < numberOfNonzeroEntries; i++)
            sum += values[i];
        return sum;
    }

    /** {@inheritDoc} */
    @Override
    public double norm(VectorNorm normType) {
        return normType.compute(Arrays.copyOf(values, numberOfNonzeroEntries));
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector add(double scalar) {
        SparseVector resultVector = new SparseVector(size, indexes, values);
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            resultVector.values[i] += scalar;
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector addInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            values[i] += scalar;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector add(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            // TODO: Perform the sparse vector casting earlier and only once.
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] =
                            values[vector1Index] + ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector addInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index] + ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector sub(double scalar) {
        SparseVector resultVector = new SparseVector(size, numberOfNonzeroEntries, indexes, values);
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            resultVector.values[i] -= scalar;
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector subInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            values[i] -= scalar;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector sub(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index] - ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector subInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index] - ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multElementwise(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index] * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index] * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divElementwise(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index] / ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divElementwiseInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index] / ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector mult(double scalar) {
        SparseVector resultVector = new SparseVector(size, numberOfNonzeroEntries, indexes, values);
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            resultVector.values[i] *= scalar;
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            values[i] *= scalar;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector div(double scalar) {
        SparseVector resultVector = new SparseVector(size, numberOfNonzeroEntries, indexes, values);
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            resultVector.values[i] /= scalar;
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++)
            values[i] /= scalar;
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector saxpy(double scalar, Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] =
                            values[vector1Index] + scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector saxpyInPlace(double scalar, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] =
                            values[vector1Index] + scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector saxpyPlusConstant(double scalar, Vector vector) {
        if (vector.size() + 1 != this.size())
            throw new IllegalArgumentException("The provided vector size must be 1 less than the current vector size.");
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] =
                            values[vector1Index] + scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector saxpyPlusConstantInPlace(double scalar, Vector vector) {
        if (vector.size() + 1 != this.size())
            throw new IllegalArgumentException("The provided vector size must be 1 less than the current vector size.");
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] =
                            values[vector1Index] + scalar * ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = scalar * ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public double inner(Vector vector) {
        checkVectorSize(vector);
        double result = 0;
        if (vector.type() == VectorType.SPARSE) {
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    result += values[vector1Index] * ((SparseVector) vector).values[vector2Index];
                    vector1Index++;
                    vector2Index++;
                }
            }
        } else {
            throw new UnsupportedOperationException();
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public double innerPlusConstant(Vector vector) {
        if (vector.size() + 1 != this.size())
            throw new IllegalArgumentException("The provided vector size must be 1 less than the current vector size.");
        double result = 0;
        if (vector.type() == VectorType.SPARSE) {
            int vector1Index = 0;
            int vector2Index = 0;
            boolean constantAdded = false;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] == size - 1) {
                    result += values[vector1Index++];
                    constantAdded = true;
                } else if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    result += values[vector1Index++] * ((SparseVector) vector).values[vector2Index++];
                }
            }
            if (!constantAdded && numberOfNonzeroEntries > 0 && indexes[numberOfNonzeroEntries - 1] == size - 1)
                result += values[numberOfNonzeroEntries - 1];
        } else {
            throw new UnsupportedOperationException();
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector hypotenuse(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = MathUtilities.computeHypotenuse(
                            values[vector1Index],
                            ((SparseVector) vector).values[vector2Index]
                    );
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector hypotenuseInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = MathUtilities.computeHypotenuse(
                            values[vector1Index],
                            ((SparseVector) vector).values[vector2Index]
                    );
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector hypotenuseFast(Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = Math.sqrt(
                            values[vector1Index] * values[vector1Index]
                                    + ((SparseVector) vector).values[vector2Index]
                                    * ((SparseVector) vector).values[vector2Index]
                    );
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector hypotenuseFastInPlace(Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = values[vector1Index];
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = Math.sqrt(
                            values[vector1Index] * values[vector1Index]
                                    + ((SparseVector) vector).values[vector2Index]
                                    * ((SparseVector) vector).values[vector2Index]
                    );
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = values[vector1Index];
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and returns it in a new
     * vector. The function is assumed to return 0 when applied to 0. Therefore, it is only applied to the nonzero
     * elements of this vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @return              A new vector holding the result of the operation.
     */
    @Override
    public SparseVector map(Function<Double, Double> function) {
        int[] newIndexes = new int[numberOfNonzeroEntries];
        double[] newValues = new double[numberOfNonzeroEntries];
        int numberOfSkippedValues = 0;
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            double tempValue = function.apply(values[i]);
            if (Math.abs(tempValue) >= epsilon) {
                newIndexes[i - numberOfSkippedValues] = indexes[i];
                newValues[i - numberOfSkippedValues] = tempValue;
            } else {
                numberOfSkippedValues++;
            }
        }
        return new SparseVector(size, numberOfNonzeroEntries - numberOfSkippedValues, newIndexes, newValues);
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and replaces the current
     * vector with the result. The function is assumed to return 0 when applied to 0. Therefore, it is only applied to
     * the nonzero elements of this vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @return              The current vector holding the result of the operation.
     */
    @Override
    public SparseVector mapInPlace(Function<Double, Double> function) {
        int[] newIndexes = new int[numberOfNonzeroEntries];
        double[] newValues = new double[numberOfNonzeroEntries];
        int numberOfSkippedValues = 0;
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            double tempValue = function.apply(values[i]);
            if (Math.abs(tempValue) >= epsilon) {
                newIndexes[i - numberOfSkippedValues] = indexes[i];
                newValues[i - numberOfSkippedValues] = tempValue;
            } else {
                numberOfSkippedValues++;
            }
        }
        numberOfNonzeroEntries -= numberOfSkippedValues;
        indexes = newIndexes;
        values = newValues;
        return this;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and the provided vector
     * (the elements of the two vectors with the same index are considered in pairs) and returns it in a new vector. The
     * function is assumed to return 0 when both of its arguments are 0. Therefore, it is only applied when either one
     * or both of the two vector values are nonzero for a each index.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to use for second argument of the function.
     * @return              A new vector holding the result of the operation.
     */
    @Override
    public SparseVector mapBiFunction(BiFunction<Double, Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = function.apply(values[vector1Index], 0.0);
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = function.apply(0.0, ((SparseVector) vector).values[vector2Index]);
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = function.apply(values[vector1Index],
                                                             ((SparseVector) vector).values[vector2Index]);
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = function.apply(values[vector1Index], 0.0);
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = function.apply(0.0, ((SparseVector) vector).values[vector2Index]);
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and the provided vector
     * (the elements of the two vectors with the same index are considered in pairs) and replaces the current vector
     * with the result. The function is assumed to return 0 when both of its arguments are 0. Therefore, it is only
     * applied when either one or both of the two vector values are nonzero for a each index.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to use for second argument of the function.
     * @return              The current vector holding the result of the operation.
     */
    @Override
    public SparseVector mapBiFunctionInPlace(BiFunction<Double, Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = function.apply(values[vector1Index], 0.0);
                    currentIndex++;
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = function.apply(0.0, ((SparseVector) vector).values[vector2Index]);
                    currentIndex++;
                    vector2Index++;
                } else {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = function.apply(values[vector1Index],
                                                             ((SparseVector) vector).values[vector2Index]);
                    currentIndex++;
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                newIndexes[currentIndex] = indexes[vector1Index];
                newValues[currentIndex] = function.apply(values[vector1Index], 0.0);
                currentIndex++;
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = function.apply(0.0, ((SparseVector) vector).values[vector2Index]);
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and adding the provided
     * vector to the result, and returns it in a new vector. The function is assumed to return 0 when applied to 0.
     * Therefore, it is only applied to the nonzero elements of this vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to add to the function result.
     * @return              A new vector holding the result of the operation.
     */
    @Override
    public SparseVector mapAdd(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index]);
                        currentIndex++;
                    }
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index])
                                + ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                    }
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                double tempValue = function.apply(values[vector1Index]);
                if (Math.abs(tempValue) >= epsilon) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = function.apply(values[vector1Index]);
                    currentIndex++;
                }
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and adding the provided
     * vector to the result, and replaces the current vector with the result. The function is assumed to return 0 when
     * applied to 0. Therefore, it is only applied to the nonzero elements of this vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to add to the function result.
     * @return              The current vector holding the result of the operation.
     */
    @Override
    public SparseVector mapAddInPlace(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index]);
                        currentIndex++;
                    }
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index])
                                + ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                    }
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                double tempValue = function.apply(values[vector1Index]);
                if (Math.abs(tempValue) >= epsilon) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = function.apply(values[vector1Index]);
                    currentIndex++;
                }
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and subtracting the
     * provided vector from the result, and returns it in a new vector. The function is assumed to return 0 when applied
     * to 0. Therefore, it is only applied to the nonzero elements of this vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to subtract from the function result.
     * @return              A new vector holding the result of the operation.
     */
    @Override
    public SparseVector mapSub(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index]);
                        currentIndex++;
                    }
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index])
                                - ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                    }
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                double tempValue = function.apply(values[vector1Index]);
                if (Math.abs(tempValue) >= epsilon) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = function.apply(values[vector1Index]);
                    currentIndex++;
                }
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and subtracting the
     * provided vector from the result, and replaces the current vector with the result. The function is assumed to
     * return 0 when applied to 0. Therefore, it is only applied to the nonzero elements of this vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to subtract from the function result.
     * @return              The current vector holding the result of the operation.
     */
    @Override
    public SparseVector mapSubInPlace(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index]);
                        currentIndex++;
                    }
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                    newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                    currentIndex++;
                    vector2Index++;
                } else {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index])
                                - ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                    }
                    vector1Index++;
                    vector2Index++;
                }
            }
            while (vector1Index < numberOfNonzeroEntries) {
                double tempValue = function.apply(values[vector1Index]);
                if (Math.abs(tempValue) >= epsilon) {
                    newIndexes[currentIndex] = indexes[vector1Index];
                    newValues[currentIndex] = function.apply(values[vector1Index]);
                    currentIndex++;
                }
                vector1Index++;
            }
            while (vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                newIndexes[currentIndex] = ((SparseVector) vector).indexes[vector2Index];
                newValues[currentIndex] = - ((SparseVector) vector).values[vector2Index];
                currentIndex++;
                vector2Index++;
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and multiplying the
     * provided vector with the result element-wise, and returns it in a new vector. The function is assumed to return 0
     * when applied to 0. Therefore, it is only applied to the nonzero elements of this vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to multiply with the function result element-wise.
     * @return              A new vector holding the result of the operation.
     */
    @Override
    public SparseVector mapMultElementwise(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index])
                                * ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                    }
                    vector1Index++;
                    vector2Index++;
                }
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and multiplying the
     * provided vector with the result element-wise, and replaces the current vector with the result. The function is
     * assumed to return 0 when applied to 0. Therefore, it is only applied to the nonzero elements of this vector.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to multiply with the function result element-wise.
     * @return              The current vector holding the result of the operation.
     */
    @Override
    public SparseVector mapMultElementwiseInPlace(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index])
                                * ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                    }
                    vector1Index++;
                    vector2Index++;
                }
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and dividing the
     * provided vector with the result element-wise, and returns it in a new vector. The function is assumed to return 0
     * when applied to 0. Therefore, it is only applied to the nonzero elements of this vector. Cases where we have
     * division by 0 are not considered. For those cases the returned result is 0 and so the user of this class
     * should take care to avoid such cases on his own.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to divide with the function result element-wise.
     * @return              A new vector holding the result of the operation.
     */
    @Override
    public SparseVector mapDivElementwise(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        SparseVector resultVector;
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index])
                                / ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                    }
                    vector1Index++;
                    vector2Index++;
                }
            }
            resultVector = new SparseVector(size, currentIndex, newIndexes, newValues);
        } else {
            throw new UnsupportedOperationException();
        }
        return resultVector;
    }

    /**
     * Computes the result of applying the supplied function element-wise to the current vector and dividing the
     * provided vector with the result element-wise, and replaces the current vector with the result. The function is
     * assumed to return 0 when applied to 0. Therefore, it is only applied to the nonzero elements of this vector.
     * Cases where we have division by 0 are not considered. For those cases the returned result is 0 and so the user of
     * this class should take care to avoid such cases on his own.
     *
     * @param   function    The function to apply to the current vector element-wise.
     * @param   vector      The vector to divide with the function result element-wise.
     * @return              The current vector holding the result of the operation.
     */
    @Override
    public SparseVector mapDivElementwiseInPlace(Function<Double, Double> function, Vector vector) {
        checkVectorSize(vector);
        if (vector.type() == VectorType.SPARSE) {
            int[] newIndexes = new int[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            double[] newValues = new double[numberOfNonzeroEntries + ((SparseVector) vector).numberOfNonzeroEntries];
            int currentIndex = 0;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries
                    && vector2Index < ((SparseVector) vector).numberOfNonzeroEntries) {
                if (indexes[vector1Index] < ((SparseVector) vector).indexes[vector2Index]) {
                    vector1Index++;
                } else if (indexes[vector1Index] > ((SparseVector) vector).indexes[vector2Index]) {
                    vector2Index++;
                } else {
                    double tempValue = function.apply(values[vector1Index]);
                    if (Math.abs(tempValue) >= epsilon) {
                        newIndexes[currentIndex] = indexes[vector1Index];
                        newValues[currentIndex] = function.apply(values[vector1Index])
                                / ((SparseVector) vector).values[vector2Index];
                        currentIndex++;
                    }
                    vector1Index++;
                    vector2Index++;
                }
            }
            indexes = newIndexes;
            values = newValues;
            numberOfNonzeroEntries = currentIndex;
        } else {
            throw new UnsupportedOperationException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public Matrix outer(Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector gaxpy(Matrix matrix, Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector gaxpyInPlace(Matrix matrix, Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector transMult(Matrix matrix) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector prepend(double value) {
        size += 1;
        numberOfNonzeroEntries += 1;
        int[] temporaryIndexes = indexes;
        double[] temporaryValues = values;
        indexes = new int[numberOfNonzeroEntries];
        values = new double[numberOfNonzeroEntries];
        indexes[0] = 0;
        values[0] = value;
        for (int i = 1; i < numberOfNonzeroEntries; i++) {
            indexes[i] = temporaryIndexes[i - 1] + 1;
            values[i] = temporaryValues[i - 1];
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector append(double value) {
        size += 1;
        numberOfNonzeroEntries += 1;
        int[] temporaryIndexes = indexes;
        double[] temporaryValues = values;
        indexes = new int[numberOfNonzeroEntries];
        values = new double[numberOfNonzeroEntries];
        for (int i = 0; i < numberOfNonzeroEntries - 1; i++) {
            indexes[i] = temporaryIndexes[i];
            values[i] = temporaryValues[i];
        }
        indexes[numberOfNonzeroEntries - 1] = size - 1;
        values[numberOfNonzeroEntries - 1] = value;
        return this;
    }

    /**
     * This method compacts this vector. Compacting in this case is the process of removing elements that are
     * effectively equal to 0 (i.e., absolute value \(<\epsilon\), where \(\epsilon\) is the square root of the smallest
     * possible value that can be represented by a double precision floating point number) from the two parallel arrays
     * used to store the sparse vector internally.
     *
     * @return  The current vector after the compacting operation is completed.
     */
    public SparseVector compact() {
        int[] newIndexes = new int[numberOfNonzeroEntries];
        double[] newValues = new double[numberOfNonzeroEntries];
        int numberOfSkippedValues = 0;
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            if (Math.abs(values[i]) >= epsilon) {
                newIndexes[i - numberOfSkippedValues] = indexes[i];
                newValues[i - numberOfSkippedValues] = values[i];
            } else {
                numberOfSkippedValues++;
            }
        }
        numberOfNonzeroEntries -= numberOfSkippedValues;
        indexes = Arrays.copyOfRange(newIndexes, 0, numberOfNonzeroEntries);
        values = Arrays.copyOfRange(newValues, 0, numberOfNonzeroEntries);
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object object) {
        if (!(object instanceof Vector))
            return false;
        if (object == this)
            return true;

        SparseVector that = (SparseVector) object;

        if (size != that.size)
            return false;

        int vector1Index = 0;
        int vector2Index = 0;
        while (vector1Index < numberOfNonzeroEntries && vector2Index < that.numberOfNonzeroEntries) {
            if (indexes[vector1Index] < that.indexes[vector2Index]) {
                if (Math.abs(values[vector1Index]) >= epsilon)
                    return false;
                vector1Index++;
            } else if (indexes[vector1Index] > that.indexes[vector2Index]) {
                if (Math.abs(that.values[vector1Index]) >= epsilon)
                    return false;
                vector2Index++;
            } else {
                if (Math.abs(values[vector1Index] - that.values[vector1Index]) >= epsilon)
                    return false;
                vector1Index++;
                vector2Index++;
            }
        }
        while (vector1Index < numberOfNonzeroEntries) {
            if (Math.abs(values[vector1Index]) >= epsilon)
                return false;
            vector1Index++;
        }
        while (vector2Index < that.numberOfNonzeroEntries) {
            if (Math.abs(that.values[vector1Index]) >= epsilon)
                return false;
            vector2Index++;
        }

        return true;
    }

    /** {@inheritDoc} */
    @Override
    public void write(OutputStream outputStream, boolean includeType) throws IOException {
        if (includeType)
            UnsafeSerializationUtilities.writeInt(outputStream, type().ordinal());
        UnsafeSerializationUtilities.writeInt(outputStream, size);
        UnsafeSerializationUtilities.writeInt(outputStream, numberOfNonzeroEntries);
        UnsafeSerializationUtilities.writeIntArray(outputStream, indexes, numberOfNonzeroEntries);
        UnsafeSerializationUtilities.writeDoubleArray(outputStream, values, numberOfNonzeroEntries);
    }

    /**
     * Deserializes the sparse vector stored in the provided input stream and returns it.
     *
     * @param   inputStream Input stream from which the dense vector will be "read".
     * @param   includeType Boolean value indicating whether the type of the vector is to also be read from the input
     *                      stream.
     * @return              The sparse vector obtained from the provided input stream.
     * @throws  IOException
     */
    public static SparseVector read(InputStream inputStream, boolean includeType) throws IOException {
        if (includeType) {
            VectorType storedVectorType = VectorType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
            if (storedVectorType != VectorType.SPARSE)
                throw new InvalidObjectException("The stored vector is of type " + storedVectorType.name() + "!");
        }
        int size = UnsafeSerializationUtilities.readInt(inputStream);
        int numberOfNonzeroEntries = UnsafeSerializationUtilities.readInt(inputStream);
        int[] indexes = UnsafeSerializationUtilities.readIntArray(inputStream,
                                                                  numberOfNonzeroEntries,
                                                                  Math.min(numberOfNonzeroEntries, 1024 * 1024));
        double[] values = UnsafeSerializationUtilities.readDoubleArray(inputStream,
                                                                       numberOfNonzeroEntries,
                                                                       Math.min(numberOfNonzeroEntries, 1024 * 1024));
        return new SparseVector(size, indexes, values);
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getEncoder(boolean includeType) {
        return new Encoder(includeType);
    }

    /**
     * Encoder class for sparse vectors. This class extends the Java {@link InputStream} class and can be used to copy
     * sparse vector instances into other locations (e.g., in a database). Note that this encoder uses the underlying
     * vector and so, if that vector is changed, the output of this encoder might be changed and even become corrupt.
     *
     * The sparse vector is serialized in the following way: (i) the size of the vector is encoded first, (ii) the
     * number of nonzero elements in the sparse vector is encoded next, (iii) the elements of the underlying array
     * holding the indexes of the nonzero elements are encoded next, in the order in which they appear in the array, and
     * (iv) the elements of the underlying array holding the values of the nonzero elements are encoded finally, in the
     * order in which they appear in the array.
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

        /** The memory address offset of {@link #size} from the base address of the sparse vector instance that is being
         * encoded. */
        final long sizeFieldOffset;
        /** The memory address offset of {@link #numberOfNonzeroEntries} from the base address of the sparse vector
         * instance that is being encoded. */
        final long numberOfNonzeroEntriesFieldOffset;
        /** The {@link VectorType} ordinal number of the type of the vector being encoded
         * (i.e., {@link VectorType#SPARSE}). */
        final int type;
        /** Boolean value indicating whether or not to also encode the type of the current vector
         * (i.e., {@link VectorType#SPARSE}). */
        final boolean includeType;

        /** Constructs an encoder object from the current vector. */
        public Encoder(boolean includeType) {
            long typeFieldOffset;
            try {
                sizeFieldOffset = UNSAFE.objectFieldOffset(SparseVector.class.getDeclaredField("size"));
                numberOfNonzeroEntriesFieldOffset =
                        UNSAFE.objectFieldOffset(SparseVector.class.getDeclaredField("numberOfNonzeroEntries"));
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
            type = VectorType.SPARSE.ordinal();
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
                        position = numberOfNonzeroEntriesFieldOffset;
                        endPosition = numberOfNonzeroEntriesFieldOffset + 4;
                        state = EncoderState.NUMBER_OF_NONZERO_ENTRIES;
                    } else {
                        return UNSAFE.getByte(SparseVector.this, position++);
                    }
                case NUMBER_OF_NONZERO_ENTRIES:
                    if (position == endPosition) {
                        position = INT_ARRAY_OFFSET;
                        endPosition = INT_ARRAY_OFFSET + (numberOfNonzeroEntries << 2);
                        state = EncoderState.INDEXES;
                    } else {
                        return UNSAFE.getByte(SparseVector.this, position++);
                    }
                case INDEXES:
                    if (position == endPosition) {
                        position = DOUBLE_ARRAY_OFFSET;
                        endPosition = DOUBLE_ARRAY_OFFSET + (numberOfNonzeroEntries << 3);
                        state = EncoderState.VALUES;
                    } else {
                        return UNSAFE.getByte(indexes, position++);
                    }
                case VALUES:
                    if (position == endPosition)
                        return -1;
                    else
                        return UNSAFE.getByte(values, position++);
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
                    bytesRead = readBytes(SparseVector.this, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = numberOfNonzeroEntriesFieldOffset;
                        endPosition = numberOfNonzeroEntriesFieldOffset + 4;
                        state = EncoderState.NUMBER_OF_NONZERO_ENTRIES;
                    }
                case NUMBER_OF_NONZERO_ENTRIES:
                    bytesRead = readBytes(SparseVector.this, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = INT_ARRAY_OFFSET;
                        endPosition = INT_ARRAY_OFFSET + (numberOfNonzeroEntries << 2);
                        state = EncoderState.INDEXES;
                    }
                case INDEXES:
                    bytesRead = readBytes(indexes, destination, offset, length);
                    if (bytesRead != -1) {
                        return bytesRead;
                    } else {
                        position = DOUBLE_ARRAY_OFFSET;
                        endPosition = DOUBLE_ARRAY_OFFSET + (numberOfNonzeroEntries << 3);
                        state = EncoderState.VALUES;
                    }
                case VALUES:
                    return readBytes(values, destination, offset, length);
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
        /** Represents the state the encoder is in, while encoding the type of the sparse vector. */
        TYPE,
        /** Represents the state the encoder is in, while encoding the size of the sparse vector. */
        SIZE,
        /** Represents the state the encoder is in, while encoding the number of nonzero entries of the sparse
         * vector. */
        NUMBER_OF_NONZERO_ENTRIES,
        /** Represents the state the encoder is in, while encoding the underlying indexes array of the sparse vector. */
        INDEXES,
        /** Represents the state the encoder is in, while encoding the underlying values array of the sparse vector. */
        VALUES
    }
}
