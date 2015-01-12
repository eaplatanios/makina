package org.platanios.learn.data;

import org.platanios.learn.math.statistics.StatisticsUtilities;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class MultiViewDataSetInMemory<D extends MultiViewDataInstance> implements MultiViewDataSet<D> {
    List<D> dataInstances;

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
        return new MultiViewDataSetInMemory<>(dataInstances.subList(fromIndex, toIndex));
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
                currentIndex += batchSize;
                if (currentIndex >= dataInstances.size())
                    currentIndex = dataInstances.size();
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
        return new Iterator<List<D>>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return true;
            }

            @Override
            public List<D> next() {
                if (sampleWithReplacement || currentIndex + batchSize >= dataInstances.size()) {
                    StatisticsUtilities.shuffle(dataInstances);
                    currentIndex = 0;
                }
                int fromIndex = currentIndex;
                currentIndex += batchSize;
                if (currentIndex >= dataInstances.size())
                    currentIndex = dataInstances.size();
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
}
