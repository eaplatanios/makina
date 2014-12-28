package org.platanios.learn.data;

import java.util.Iterator;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface DataSet<D extends DataInstanceWithFeatures> extends Iterable<D> {
    public int size();
    public void add(D dataInstance);
    public D get(int index);
    public DataSet<D> subSet(int fromIndex, int toIndex);
    @Override
    public Iterator<D> iterator();
    public Iterator<List<D>> batchIterator(int batchSize);
    public Iterator<List<D>> continuousRandomBatchIterator(int batchSize, boolean sampleWithReplacement);
}
