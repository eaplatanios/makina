package org.platanios.learn.data;

import java.util.Iterator;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface MultiViewDataSet<D extends MultiViewDataInstance> extends Iterable<D> {
    public int size();
    public void add(D dataInstance);
    public D get(int index);
    public MultiViewDataSet<D> subSet(int fromIndex, int toIndex);
    @Override
    public Iterator<D> iterator();
    public Iterator<List<D>> batchIterator(int batchSize);
    public Iterator<List<D>> continuousRandomBatchIterator(int batchSize, boolean sampleWithReplacement);
}
