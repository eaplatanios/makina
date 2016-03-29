package org.platanios.learn.data;

import org.platanios.math.StatisticsUtilities;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MultiViewDataSetInMemory<D extends MultiViewDataInstance> implements MultiViewDataSet<D> {
    private List<D> dataInstances;

    public MultiViewDataSetInMemory() {
        this.dataInstances = new ArrayList<>();
    }

    public MultiViewDataSetInMemory(List<D> dataInstances) {
        this.dataInstances = dataInstances;
    }

    @Override
    public int size() {
        return dataInstances.size();
    }

    @Override
    public <S extends MultiViewDataInstance> MultiViewDataSet<S> newDataSet() {
        return new MultiViewDataSetInMemory<>();
    }

    @Override
    public void add(D dataInstance) {
        dataInstances.add(dataInstance);
    }

    @Override
    public void add(List<D> dataInstances) {
        dataInstances.addAll(dataInstances);
    }

    @Override
    public void remove(int index) {
        dataInstances.remove(index);
    }

    @Override
    public D get(int index) {
        return dataInstances.get(index);
    }

    @Override
    public void set(int index, D dataInstance) {
        dataInstances.set(index, dataInstance);
    }

    @Override
    public MultiViewDataSetInMemory<D> subSet(int fromIndex, int toIndex) {
        return new MultiViewDataSetInMemory<>(new ArrayList<>(dataInstances.subList(fromIndex, toIndex)));
    }

    @Override
    public MultiViewDataSetInMemory<D> subSetComplement(int fromIndex, int toIndex) {
        List<D> dataInstancesList = new ArrayList<>(dataInstances.subList(0, fromIndex));
        dataInstancesList.addAll(dataInstances.subList(toIndex, dataInstances.size()));
        return new MultiViewDataSetInMemory<>(dataInstancesList);
    }

    @Override
    public DataSetInMemory<? extends DataInstance> getSingleViewDataSet(int view) {
        return new DataSetInMemory<>(dataInstances
                                             .parallelStream()
                                             .map(dataInstance -> dataInstance.getSingleViewDataInstance(view))
                                             .collect(Collectors.toList()));
    }

    @Override
    public MultiViewDataSetInMemory<D> sort(Comparator<? super D> comparator) {
        dataInstances.sort(comparator);
        return this;
    }

    @Override
    public Iterator<D> iterator() {
        return new Iterator<D>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < dataInstances.size();
            }

            @Override
            public D next() {
                return dataInstances.get(currentIndex++);
            }

            @Override
            public void remove() {
                dataInstances.remove(--currentIndex);
            }
        };
    }

    @Override
    public Iterator<List<D>> batchIterator(int batchSize) {
        return new Iterator<List<D>>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < dataInstances.size();
            }

            @Override
            public List<D> next() {
                int fromIndex = currentIndex;
                currentIndex = Math.min(currentIndex + batchSize, dataInstances.size());
                return dataInstances.subList(fromIndex, currentIndex);
            }

            @Override
            public void remove() {
                currentIndex--;
                int indexLowerBound = currentIndex - batchSize;
                while (currentIndex > indexLowerBound)
                    dataInstances.remove(currentIndex--);
            }
        };
    }

    @Override
    public Iterator<List<D>> continuousRandomBatchIterator(int batchSize, boolean sampleWithReplacement) {
        return continuousRandomBatchIterator(batchSize, sampleWithReplacement, null);
    }

    @Override
    public Iterator<List<D>> continuousRandomBatchIterator(int batchSize,
                                                           boolean sampleWithReplacement,
                                                           Random random) {
        final List<Integer> dataInstancesIndexes = new ArrayList<>(dataInstances.size());
        for (int i = 0; i < dataInstances.size(); i++)
            dataInstancesIndexes.add(i);

        return new Iterator<List<D>>() {
            private int currentIndex = 0;
            private List<Integer> indexes = dataInstancesIndexes;

            @Override
            public boolean hasNext() {
                return true;
            }

            @Override
            public List<D> next() {
                if (sampleWithReplacement || currentIndex + batchSize >= dataInstances.size()) {
                    if (random == null)
                        StatisticsUtilities.shuffle(indexes);
                    else
                        StatisticsUtilities.shuffle(indexes, random);
                    currentIndex = 0;
                }
                int fromIndex = currentIndex;
                currentIndex = Math.min(currentIndex + batchSize, dataInstances.size());
                List<D> dataInstancesSubList = new ArrayList<>(batchSize);
                for (int i = fromIndex; i < currentIndex; i++)
                    dataInstancesSubList.add(dataInstances.get(indexes.get(i)));
                return dataInstancesSubList;
            }

            @Override
            public void remove() {
                currentIndex--;
                int indexLowerBound = currentIndex - batchSize;
                while (currentIndex > indexLowerBound) {
                    dataInstancesIndexes.remove(indexes.get(currentIndex));
                    dataInstances.remove((int) indexes.remove(currentIndex--));
                }
            }
        };
    }
}
