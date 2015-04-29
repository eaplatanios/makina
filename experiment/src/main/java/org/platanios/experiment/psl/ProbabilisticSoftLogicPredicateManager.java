package org.platanios.experiment.psl;

import java.util.*;

/**
 * Created by Dan on 4/26/2015.
 * Manages the ids of entities and predicates, along with some meta information
 */
public class ProbabilisticSoftLogicPredicateManager {

    public ProbabilisticSoftLogicReader.Predicate getPredicateFromId(int id) {
        return this.idToPredicate.getOrDefault(id, null);
    }

    public double getObservedWeight(ProbabilisticSoftLogicReader.Predicate predicate) {
        return this.getObservedWeight(getIdForPredicate(predicate));
    }

    public double getObservedWeight(int id) {
        return this.observedPredicateWeights.getOrDefault(id, Double.NaN);
    }

    public List<AbstractMap.SimpleEntry<Integer, Double>> getObservedIndices(int[] ... allIndices) {

        ArrayList<AbstractMap.SimpleEntry<Integer, Double>> result = new ArrayList<>();
        for (int[] list : allIndices) {
            for (int id : list) {
                double weight = this.getObservedWeight(id);
                if (!Double.isNaN(weight)) {
                    result.add(new AbstractMap.SimpleEntry<Integer, Double>(id, weight));
                }
            }
        }

        return result;
    }

    public int getOrAddPredicate(ProbabilisticSoftLogicReader.Predicate predicate) {
        return this.getOrAddPredicate(predicate, Double.NaN);
    }

    public int getOrAddPredicate(ProbabilisticSoftLogicReader.Predicate predicate, double observedWeight) {

        int id = this.predicateToId.getOrDefault(predicate.toString(), -1);
        if (id >= 0) {

            double currentWeight = this.observedPredicateWeights.getOrDefault(id, Double.NaN);
            if (Double.isNaN(currentWeight)) {
                if (!Double.isNaN(observedWeight))
                    throw new UnsupportedOperationException("Current value of observedWeight mismatches weight being added");
            } else {
                if (currentWeight != observedWeight) {
                    throw new UnsupportedOperationException("Current value of observedWeight mismatches weight being added");
                }
            }

            return id;

        }

        ArrayList<Integer> groundingsForName = this.predicateNameToIds.getOrDefault(predicate.Name, null);
        if (groundingsForName == null) {
            groundingsForName = new ArrayList<>();
            this.predicateNameToIds.put(predicate.Name, groundingsForName);
        }
        groundingsForName.add(this.nextId);

        ArrayList<HashSet<String>> argumentGroundingsForName = this.argumentGroundings.getOrDefault(predicate.Name, null);
        if (argumentGroundingsForName == null) {
            argumentGroundingsForName = new ArrayList<>();
            this.argumentGroundings.put(predicate.Name, argumentGroundingsForName);
        }

        for (int i = 0; i < predicate.Arguments.size(); ++i) {
            if (i >= argumentGroundingsForName.size()) {
                argumentGroundingsForName.add(new HashSet<>());
            }
            argumentGroundingsForName.get(i).add(predicate.Arguments.get(i));
        }

        this.predicateToId.put(predicate.toString(), this.nextId);
        this.idToPredicate.put(nextId, predicate);

        if (!Double.isNaN(observedWeight)) {
            this.observedPredicateWeights.put(this.nextId, observedWeight);
        }

        id = this.nextId++;

        return id;
    }

    public int getIdForPredicate(ProbabilisticSoftLogicReader.Predicate predicate) {
        return this.predicateToId.getOrDefault(predicate.toString(), -1);
    }

    public List<Integer> getIdsForPredicateName(String predicateName) {
        return this.predicateNameToIds.getOrDefault(predicateName, new ArrayList<Integer>());
    }

    public Iterator<String> getArgumentGroundings(String predicateName, int argumentPosition) {

        ArrayList<HashSet<String>> argumentGroundingsForName = this.argumentGroundings.getOrDefault(predicateName, null);
        if (argumentGroundingsForName == null || argumentGroundingsForName.size() <= argumentPosition) {
            return (new HashSet<String>()).iterator();
        }

        return argumentGroundingsForName.get(argumentPosition).iterator();

    }

    private int nextId = 0;
    private final HashMap<String, Integer> predicateToId = new HashMap<>();
    private final HashMap<String, ArrayList<Integer>> predicateNameToIds = new HashMap<>();
    private final HashMap<String, ArrayList<HashSet<String>>> argumentGroundings = new HashMap<>();
    private final HashMap<Integer, ProbabilisticSoftLogicReader.Predicate> idToPredicate = new HashMap<>();
    private final HashMap<Integer, Double> observedPredicateWeights = new HashMap<>();

}
