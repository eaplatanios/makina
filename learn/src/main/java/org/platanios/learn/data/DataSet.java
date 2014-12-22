package org.platanios.learn.data;

import java.util.Iterator;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface DataSet<D extends DataInstanceBase> extends Iterable<D> {
    public void add(D dataInstance);
    @Override
    public Iterator<D> iterator();
    public Iterator<List<D>> batchIterator(int batchSize);
    public Iterator<List<D>> continuousRandomBatchIterator(int batchSize, boolean sampleWithReplacement);
}
