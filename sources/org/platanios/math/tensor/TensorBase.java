package org.platanios.math.tensor;

/**
 * Common methods shared by tensor classes. These methods define a common access mechanism across both mutable and
 * immutable variants of tensors.
 *
 * @author Emmanouil Antonios Platanios
 */
public interface TensorBase {
    int getNumberOfDimensions();
    int[] getSize();
    int getNumberOfElements();
    double getElementByKey(long key);
    double getElementByIndex(int[] index);
    int[] convertKeyToIndex(long key);
    long convertIndexToKey(int[] index);
    double getTrace();
    double getL2Norm();
    long[] getLargestValues(int n);
}
