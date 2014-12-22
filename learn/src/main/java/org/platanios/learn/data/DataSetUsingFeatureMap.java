package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.statistics.StatisticsUtilities;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @param   <T>
 * @param   <D>
 * @param   <F>
 *
 * @author Emmanouil Antonios Platanios
 */
public class DataSetUsingFeatureMap<T extends Vector, D extends F, F extends DataInstanceBase<T>> implements DataSet<D> {
    final FeatureMap<T> featureMap;
    final int featureMapView;
    List<F> dataInstances;

    public DataSetUsingFeatureMap(FeatureMap<T> featureMap, int featureMapView) {
        this.featureMap = featureMap;
        this.featureMapView = featureMapView;
        this.dataInstances = new ArrayList<>();
    }

    public DataSetUsingFeatureMap(FeatureMap<T> featureMap, int featureMapView, List<F> dataInstances) {
        this.featureMap = featureMap;
        this.featureMapView = featureMapView;
        this.dataInstances = new ArrayList<>(dataInstances);
    }

    @Override
    public int size() {
        return dataInstances.size();
    }

    @Override
    @SuppressWarnings("unchecked")
    public void add(D dataInstance) {
        dataInstances.add((F) dataInstance.toDataInstanceBase());
    }

    @Override
    @SuppressWarnings("unchecked")
    public D get(int index) {
        F dataInstance = dataInstances.get(index);
        return (D) dataInstance.toDataInstance(featureMap.getFeatureVector(dataInstance.name(), featureMapView));
    }

    @Override
    public DataSetUsingFeatureMap<T, D, F> subSet(int fromIndex, int toIndex) {
        return new DataSetUsingFeatureMap<>(featureMap, featureMapView, dataInstances.subList(fromIndex, toIndex));
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
            @SuppressWarnings("unchecked")
            public D next() {
                F dataInstance = dataInstances.get(currentIndex++);
                return (D) dataInstance.toDataInstance(featureMap.getFeatureVector(dataInstance.name(),
                                                                                   featureMapView));
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
            @SuppressWarnings("unchecked")
            public List<D> next() {
                int fromIndex = currentIndex;
                currentIndex += batchSize;
                if (currentIndex >= dataInstances.size())
                    currentIndex = dataInstances.size();
                List<F> dataInstancesSubList = dataInstances.subList(fromIndex, currentIndex);
                return dataInstancesSubList.stream()
                        .map(dataInstance ->
                                     (D) dataInstance.toDataInstance(
                                             featureMap.getFeatureVector(dataInstance.name(), featureMapView)
                                     ))
                        .collect(Collectors.toList());
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
            @SuppressWarnings("unchecked")
            public List<D> next() {
                if (sampleWithReplacement || currentIndex + batchSize >= dataInstances.size()) {
                    StatisticsUtilities.shuffle(dataInstances);
                    currentIndex = 0;
                }
                int fromIndex = currentIndex;
                currentIndex += batchSize;
                if (currentIndex >= dataInstances.size())
                    currentIndex = dataInstances.size();
                List<F> dataInstancesSubList = dataInstances.subList(fromIndex, currentIndex);
                return dataInstancesSubList.stream()
                        .map(dataInstance ->
                                     (D) dataInstance.toDataInstance(
                                             featureMap.getFeatureVector(dataInstance.name(), featureMapView)
                                     ))
                        .collect(Collectors.toList());
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
