package org.platanios.learn.data;

import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public interface DataSet<D extends DataInstance> extends Iterable<D> {
    public int size();
    public void add(D dataInstance);
    public void add(List<D> dataInstances);
    public void remove(int index);
    public D get(int index);
    public void set(int index, D dataInstance);
    public DataSet<D> subSet(int fromIndex, int toIndex);
    public DataSet<D> sort(Comparator<? super D> comparator);
    @Override
    public Iterator<D> iterator();
    public Iterator<List<D>> batchIterator(int batchSize);
    public Iterator<List<D>> continuousRandomBatchIterator(int batchSize, boolean sampleWithReplacement);
    public Iterator<List<D>> continuousRandomBatchIterator(int batchSize, boolean sampleWithReplacement, Random random);
}
