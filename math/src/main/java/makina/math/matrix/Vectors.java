package makina.math.matrix;

import cern.colt.map.OpenIntDoubleHashMap;
import makina.utilities.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Random;

/**
 * This class provides several static methods to build vectors of different types and initialize them in various ways.
 *
 * @author Emmanouil Antonios Platanios
 */
public class Vectors {
    private final static Random random = new Random();

    // Suppress default constructor for noninstantiability
    private Vectors() {
        throw new AssertionError();
    }

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
     * Builds a vector from the contents of the provided input stream. Note that the contents of the stream must have
     * been written using the write(InputStream, boolean) function of the corresponding vector class, with the second
     * parameter being equal to True, in order to be compatible with this constructor. If the contents are not
     * compatible, then an {@link java.io.IOException} might be thrown, or the constructed vector might be corrupted in
     * some way.
     *
     * @param   inputStream The input stream to read the contents of this vector from.
     * @return              The new vector.
     * @throws java.io.IOException
     */
    public static Vector build(InputStream inputStream) throws IOException {
        VectorType vectorType = VectorType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
        return vectorType.buildVector(inputStream, false);
    }

    // TODO: Mention that this is always dense.
    public static DenseVector ones(int size) {
        return new DenseVector(size, 1);
    }

    // TODO: Mention that this is always dense.
    public static DenseVector random(int size) {
        DenseVector vector = new DenseVector(size);
        for (int i = 0; i < vector.size(); i++)
            vector.set(i, random.nextDouble());
        return vector;
    }

    /**
     * Builds a dense vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     * @return          The new vector.
     */
    public static DenseVector dense(int size) {
        return new DenseVector(size);
    }

    /**
     * Builds a dense vector of the given size and fills it with the provided computeValue.
     *
     * @param   size    The size of the vector.
     * @param   value   The computeValue with which to fill the vector.
     * @return          The new vector.
     */
    public static DenseVector dense(int size, double value) {
        return new DenseVector(size, value);
    }

    /**
     * Builds a dense vector from a one-dimensional array.
     *
     * @param   elements    One-dimensional array of doubles.
     * @return              The new vector.
     */
    public static DenseVector dense(double... elements) {
        return new DenseVector(elements);
    }

    /**
     * Constructs a sparse vector of the given size and fills it with zeros.
     *
     * @param   size    The size of the vector.
     */
    public static SparseVector sparse(int size) {
        return new SparseVector(size);
    }

    /**
     * Constructs a sparse vector of the given size and fills it with the values stored in the provided map. The map
     * must contain key-computeValue pairs where the key corresponds to an element index and the computeValue to the corresponding
     * element's computeValue. Note that the map does not need to be an sorted map; all the necessary sorting is performed
     * within this constructor.
     *
     * @param   size            The size of the vector.
     * @param   vectorElements  The map containing the vector indexes and values used to initialize the values of the
     *                          elements of this vector.
     */
    public static SparseVector sparse(int size, Map<Integer, Double> vectorElements) {
        return new SparseVector(size, vectorElements);
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
    public static SparseVector sparse(int size, int[] indexes, double[] values) {
        return new SparseVector(size, indexes, values);
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
    public static SparseVector sparse(int size, int numberOfNonzeroEntries, int[] indexes, double[] values) {
        return new SparseVector(size, numberOfNonzeroEntries, indexes, values);
    }

    /**
     * Constructs a sparse vector from a dense vector. This constructor does not simply transform the dense vector
     * structure into a sparse vector structure, but it also throws away elements of the dense vector that have a computeValue
     * effectively 0 (i.e., absolute computeValue \(<\epsilon\), where \(\epsilon\) is the square root of the smallest possible
     * computeValue that can be represented by a double precision floating point number).
     *
     * @param   vector  The dense vector from which to construct this sparse vector.
     */
    public static SparseVector sparse(DenseVector vector) {
        return new SparseVector(vector);
    }

    /**
     * Constructs a sparse vector from another sparse vector. This constructor basically constructs a copy of the
     * provided sparse vector.
     *
     * @param   vector  The sparse vector from which to construct this sparse vector.
     */
    public static SparseVector sparse(SparseVector vector) {
        return new SparseVector(vector);
    }

    /**
     * Constructs a sparse vector from another hash vector.
     *
     * @param   vector  The hash vector from which to construct this sparse vector.
     */
    public static SparseVector sparse(HashVector vector) {
        return new SparseVector(vector);
    }

    /**
     * Builds a sparse vector of the given size containing only zeros.
     *
     * @param   size        The size of the vector.
     * @return              The new vector.
     */
    public static HashVector hash(int size) {
        return new HashVector(size);
    }

    /**
     * Builds a sparse vector of the given size from a hash map.
     *
     * @param   size        The size of the vector.
     * @param   elements    Hash map containing the indexes of elements as keys and the values of the corresponding
     *                      elements as values.
     * @return              The new vector.
     */
    public static HashVector hash(int size, OpenIntDoubleHashMap elements) {
        return new HashVector(size, elements);
    }
}
