package org.platanios.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.primitives.Ints;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A structure that holds the individual error rates of a set of functions, as well as the joint error rates of all
 * possible subsets of those functions.
 *
 * @author Emmanouil Antonios Platanios
 */
final class ErrorRatesPowerSetVector extends PowerSetVector {
    /**
     * Constructs a power set indexed vector, holding the error rates, stored as a one-dimensional array and initializes
     * the values of those error rates to be 0.25 for the individual function error rates and a computed computeValue for the
     * joint error rates assuming that each function's error rate is independent of the rest (this independence
     * assumption is made only for the initialization of those error rates values and has nothing to do with the actual
     * error rates estimation problem).
     *
     * @param   numberOfFunctions   The total number of functions considered.
     * @param   highestOrder        The highest cardinality of sets of functions to consider, out of the whole power
     *                              set.
     */
    public ErrorRatesPowerSetVector(int numberOfFunctions,
                                    int highestOrder) {
        super(numberOfFunctions, 1, highestOrder);
        initializeValues(0.25);
    }

    /**
     * Constructs a power set indexed vector, holding the error rates, stored as a one-dimensional array and initializes
     * the values of those error rates to be {@code initialValue} for the individual function error rates and a computed
     * computeValue for the joint error rates assuming that each function's error rate is independent of the rest (this
     * independence assumption is made only for the initialization of those error rates values and has nothing to do
     * with the actual error rates estimation problem).
     *
     * @param   numberOfFunctions   The total number of functions considered.
     * @param   highestOrder        The highest cardinality of sets of functions to consider, out of the whole power
     *                              set.
     * @param   initialValue        The initial computeValue for the individual function error rates.
     */
    public ErrorRatesPowerSetVector(int numberOfFunctions,
                                    int highestOrder,
                                    double initialValue) {
        this(numberOfFunctions, highestOrder);
        initializeValues(initialValue);
    }

    /**
     * Constructs a power set indexed vector, holding the error rates, stored as a one-dimensional array and initializes
     * the values of those error rates to the sample error rates computed from the available labeled data samples and
     * the outputs of the functions for each data sample.
     *
     * @param   numberOfFunctions   The total number of functions considered.
     * @param   highestOrder        The highest cardinality of sets of functions to consider, out of the whole power
     *                              set.
     */
    public ErrorRatesPowerSetVector(int numberOfFunctions,
                                    int highestOrder,
                                    Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                                    Integrator.Data<Integrator.Data.ObservedInstance> observedData,
                                    BiMap<Integer, Integer> functionIdsMap) {
        this(numberOfFunctions, highestOrder);
        computeValues(predictedData, observedData, functionIdsMap);
    }

    /**
     * Initializes the values of the function error rates to be {@code initialValue} for the individual function error
     * rates and a computed computeValue for the joint error rates assuming that each function's error rate is independent of
     * the rest (this independence assumption is made only for the initialization of those error rates values and has
     * nothing to do with the actual error rates estimation problem).
     *
     * @param   initialValue    The initial computeValue for the individual function error rates.
     */
    private void initializeValues(double initialValue) {
        for (BiMap.Entry<List<Integer>, Integer> indexKeyPair : indexKeyMapping.entrySet()) {
            List<Integer> index = indexKeyPair.getKey();
            Integer key = indexKeyPair.getValue();
            if (index.size() == 1) {
                array[key] = initialValue;
            } else {
                array[key] = 1;
                for (Integer i : index)
                    array[key] *= array[indexKeyMapping.get(Ints.asList(i))];
            }
        }
    }

    /**
     * Computes the sample error rates from the available labeled data samples and the outputs of the functions for each
     * data sample and sets the values of the function error rates to those computed values.
     */
    private void computeValues(Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                               Integrator.Data<Integrator.Data.ObservedInstance> observedData,
                               BiMap<Integer, Integer> functionIdsMap) {
        Map<Integer, Map<Integer, Boolean>> dataMap = new HashMap<>();
        predictedData.stream().forEach(
                instance -> dataMap.computeIfAbsent(instance.id(), key -> new HashMap<>())
                        .put(functionIdsMap.get(instance.functionId()), instance.value() >= 0.5)
        );
        Map<Integer, Boolean> observedDataMap = new HashMap<>();
        observedData.stream().forEach(instance -> observedDataMap.put(instance.id(), instance.value()));
        int[] counts = new int[array.length];
        for (Map.Entry<Integer, Map<Integer, Boolean>> dataMapEntry : dataMap.entrySet()) {
            if (!observedDataMap.containsKey(dataMapEntry.getKey()))
                continue;
            Map<Integer, Boolean> dataMapInstance = dataMapEntry.getValue();
            for (BiMap.Entry<List<Integer>, Integer> entry : indexKeyMapping.entrySet()) {
                boolean equal = true;
                List<Integer> indexes = entry.getKey();
                if (!dataMapInstance.keySet().containsAll(indexes))
                    continue;
                counts[entry.getValue()]++;
                for (int index : indexes.subList(1, indexes.size()))
                    equal = equal && (dataMapInstance.get(indexes.get(0)) == dataMapInstance.get(index));
                if (equal && dataMapInstance.get(indexes.get(0)) != observedDataMap.get(dataMapEntry.getKey()))
                    array[entry.getValue()]++;
            }
        }
        for (int i = 0; i < length; i++)
            array[i] /= counts[i];
    }
}
