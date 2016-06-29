package makina.learn.classification.reflection;

import com.google.common.collect.BiMap;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A structure that holds the agreement rates of all possible subsets of a set of functions of cardinality greater than
 * or equal to 2.
 *
 * @author Emmanouil Antonios Platanios
 */
final class AgreementRatesPowerSetVector extends PowerSetVector {
    /**
     * Constructs a power set indexed vector, holding the agreement rates, stored as a one-dimensional array and
     * initializes the values of those agreement rates to the sample agreement rates computed from the observed outputs
     * of the functions for a set of data samples, where only subsets of functions of even cardinality are
     * considered. That is because the agreement rates of a subset of functions with odd cardinality can be written in
     * terms of the agreement rates of all subsets of functions of even cardinality less than the original subset
     * cardinality.
     *
     * @param   numberOfFunctions   The total number of functions considered.
     * @param   highestOrder        The highest cardinality of sets of functions to consider, out of the whole power
     *                              set.
     */
    public AgreementRatesPowerSetVector(int numberOfFunctions,
                                        int highestOrder,
                                        Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                                        BiMap<Integer, Integer> functionIdsMap) {
        super(numberOfFunctions, 2, highestOrder, true);
        computeValues(predictedData, functionIdsMap);
    }

    /**
     * Constructs a power set indexed vector, holding the agreement rates, stored as a one-dimensional array and
     * initializes the values of those agreement rates to the sample agreement rates computed from the observed outputs
     * of the functions for a set of data samples, where an option is given on whether or not to only consider sets of
     * even cardinality within the power set.
     *
     * @param   numberOfFunctions           The total number of functions considered.
     * @param   highestOrder                The highest cardinality of sets of functions to consider, out of the whole
     *                                      power set.
     * @param onlyEvenCardinalitySubsets    Boolean computeValue indicating whether or not to only consider sets of even
     *                                      cardinality, out of the whole power set.
     */
    public AgreementRatesPowerSetVector(int numberOfFunctions,
                                        int highestOrder,
                                        Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                                        boolean onlyEvenCardinalitySubsets,
                                        BiMap<Integer, Integer> functionIdsMap) {
        super(numberOfFunctions, 2, highestOrder, onlyEvenCardinalitySubsets);
        computeValues(predictedData, functionIdsMap);
    }

    /**
     * Computes the sample agreement rates from the observed outputs of the functions for a set of data samples and sets
     * the values of the function agreement rates to those computed values.
     */
    private void computeValues(Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                               BiMap<Integer, Integer> functionIdsMap) {
        Map<Integer, Map<Integer, Boolean>> dataMap = new HashMap<>();
        predictedData.stream().forEach(
                instance -> dataMap.computeIfAbsent(instance.id(), key -> new HashMap<>())
                        .put(functionIdsMap.get(instance.functionId()), instance.value() >= 0.5)
        );
        int[] counts = new int[array.length];
        for (Map<Integer, Boolean> dataMapInstance : dataMap.values())
            for (BiMap.Entry<List<Integer>, Integer> entry : indexKeyMapping.entrySet()) {
                boolean equal = true;
                List<Integer> indexes = entry.getKey();
                if (!dataMapInstance.keySet().containsAll(indexes))
                    continue;
                counts[entry.getValue()]++;
                for (int index : indexes.subList(1, indexes.size()))
                    equal = equal && (dataMapInstance.get(indexes.get(0)) == dataMapInstance.get(index));
                if (equal)
                    array[entry.getValue()]++;
            }
        for (int i = 0; i < length; i++)
            array[i] /= counts[i];
    }
}
