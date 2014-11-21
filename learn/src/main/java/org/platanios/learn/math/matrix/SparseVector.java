package org.platanios.learn.math.matrix;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.platanios.learn.math.MathUtilities;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.*;
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
    private static final long serialVersionUID = -5387808766447862936L;

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
        throw new NotImplementedException();
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
        for (int i = 1; i < numberOfNonzeroEntries; i++) {
            minValue = Math.min(minValue, values[i]);
        }
        return minValue;
    }

    /** {@inheritDoc} */
    @Override
    public double sum() {
        double sum = values[0];
        for (int i = 1; i < numberOfNonzeroEntries; i++) {
            sum += values[i];
        }
        return sum;
    }

    /** {@inheritDoc} */
    @Override
    public double norm(VectorNorm normType) {
        return normType.compute(values);
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector add(double scalar) {
        SparseVector resultVector = new SparseVector(size, indexes, values);
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] += scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector addInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            values[i] += scalar;
        }
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector sub(double scalar) {
        SparseVector resultVector = new SparseVector(size, indexes, values);
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] -= scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector subInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            values[i] -= scalar;
        }
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector mult(double scalar) {
        SparseVector resultVector = new SparseVector(size, indexes, values);
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] *= scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector multInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            values[i] *= scalar;
        }
        return this;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector div(double scalar) {
        SparseVector resultVector = new SparseVector(size, indexes, values);
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            resultVector.values[i] /= scalar;
        }
        return resultVector;
    }

    /** {@inheritDoc} */
    @Override
    public SparseVector divInPlace(double scalar) {
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            values[i] /= scalar;
        }
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
            throw new NotImplementedException();
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
    public Vector gaxpy(Matrix matrix, Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Vector gaxpyInPlace(Matrix matrix, Vector vector) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Vector transMult(Matrix matrix) {
        return null;
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
    public void writeObject(ObjectOutputStream outputStream) throws IOException {
        outputStream.writeObject(type());
        outputStream.writeInt(size);
        outputStream.writeInt(numberOfNonzeroEntries);
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            outputStream.writeInt(indexes[i]);
            outputStream.writeDouble(values[i]);
        }
    }

    /** {@inheritDoc} */
    @Override
    protected void readObject(ObjectInputStream inputStream) throws IOException, ClassNotFoundException {
        VectorType storedVectorType = (VectorType) inputStream.readObject();
        if (storedVectorType != type())
            throw new InvalidObjectException("The stored vector is of type " + storedVectorType.name() + "!");
        size = inputStream.readInt();
        numberOfNonzeroEntries = inputStream.readInt();
        indexes = new int[numberOfNonzeroEntries];
        values = new double[numberOfNonzeroEntries];
        for (int i = 0; i < numberOfNonzeroEntries; i++) {
            indexes[i] = inputStream.readInt();
            values[i] = inputStream.readDouble();
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object object) {
        if (!(object instanceof Vector))
            return false;
        if (object == this)
            return true;

        if (((Vector) object).type() == VectorType.SPARSE) {
            SparseVector otherVector = (SparseVector) object;
            int vector1Index = 0;
            int vector2Index = 0;
            while (vector1Index < numberOfNonzeroEntries && vector2Index < otherVector.numberOfNonzeroEntries) {
                if (indexes[vector1Index] < otherVector.indexes[vector2Index]) {
                    if (Math.abs(values[vector1Index]) >= epsilon)
                        return false;
                    vector1Index++;
                } else if (indexes[vector1Index] > otherVector.indexes[vector2Index]) {
                    if (Math.abs(otherVector.values[vector1Index]) >= epsilon)
                        return false;
                    vector2Index++;
                } else {
                    if (Math.abs(values[vector1Index] - otherVector.values[vector1Index]) >= epsilon)
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
            while (vector2Index < otherVector.numberOfNonzeroEntries) {
                if (Math.abs(otherVector.values[vector1Index]) >= epsilon)
                    return false;
                vector2Index++;
            }
            return true;
        } else {
            throw new NotImplementedException();
        }
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 31) // Two randomly chosen prime numbers.
                .append(size)
                .append(indexes)
                .append(values)
                .toHashCode();
    }
}
