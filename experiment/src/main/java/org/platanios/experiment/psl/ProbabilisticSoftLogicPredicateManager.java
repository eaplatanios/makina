package org.platanios.experiment.psl;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import java.io.Serializable;
import java.util.*;

/**
 * Created by Dan on 4/26/2015.
 * Manages the ids of entities and predicates, along with some meta information
 */
public class ProbabilisticSoftLogicPredicateManager implements Serializable {

    public ProbabilisticSoftLogicProblem.Predicate getPredicateFromId(int id) {
        return this.predicateToId.inverse().getOrDefault(id, null);
    }

    public double getObservedWeight(int id) {
        return this.observedPredicateWeights.getOrDefault(id, Double.NaN);
    }

    public class IdWeights {

        public IdWeights(int[] ids, double[] weights) {
            this.Ids = ids;
            this.Weights = weights;
        }

        public final int[] Ids;
        public final double[] Weights;

    }

    public IdWeights getAllObservedWeights() {

        int[] ids = new int[this.observedPredicateWeights.size()];
        double[] weights = new double[this.observedPredicateWeights.size()];

        Iterator<Map.Entry<Integer, Double>> entryIterator = this.observedPredicateWeights.entrySet().iterator();
        int index = 0;
        while (entryIterator.hasNext()) {
            Map.Entry<Integer, Double> entry = entryIterator.next();
            ids[index] = entry.getKey();
            weights[index] = entry.getValue();
            ++index;
        }

        return new IdWeights(ids, weights);

    }

    public Set<Map.Entry<Integer, ProbabilisticSoftLogicProblem.Predicate>> getAllPredicates() {
        return Collections.unmodifiableSet(this.predicateToId.inverse().entrySet());
    }

    public Set<String> getAllEntities() {
        return Collections.unmodifiableSet(this.entities);
    }

    public Set<String> getPredicateNames() {
        return Collections.unmodifiableSet(this.argumentGroundings.keySet());
    }

    public Map.Entry<Boolean, Integer> getOrAddPredicate(ProbabilisticSoftLogicProblem.Predicate predicate) {
        return this.getOrAddPredicate(predicate, Double.NaN);
    }

    public Map.Entry<Boolean, Integer> getOrAddPredicate(ProbabilisticSoftLogicProblem.Predicate predicate, double observedWeight) {

        if (this.closedPredicates.contains(predicate.Name)) {
            throw new UnsupportedOperationException("Cannot add to a closed predicate set");
        }

        int id = this.predicateToId.getOrDefault(predicate, -1);
        if (id >= 0) {

            double currentWeight = this.observedPredicateWeights.getOrDefault(id, Double.NaN);
            if (Double.isNaN(currentWeight)) {
                if (!Double.isNaN(observedWeight))
                    throw new UnsupportedOperationException("Current value of observedWeight mismatches weight being added");
            } else {
                if (!Double.isNaN(observedWeight)) {
                    if (currentWeight != observedWeight) {
                        throw new UnsupportedOperationException("Current value of observedWeight mismatches weight being added");
                    }
                }
            }

            return new AbstractMap.SimpleEntry<>(false, id);

        }

        ArrayList<HashSet<String>> groundingsForName = this.argumentGroundings.getOrDefault(predicate.Name, null);
        if (groundingsForName == null) {
            groundingsForName = new ArrayList<>();
            this.argumentGroundings.put(predicate.Name, groundingsForName);
        }

        for (int indexArgument = 0; indexArgument < predicate.Arguments.size(); ++indexArgument) {

            if (indexArgument >= groundingsForName.size()) {
                groundingsForName.add(new HashSet<>());
            }

            groundingsForName.get(indexArgument).add(predicate.Arguments.get(indexArgument));
            this.entities.add(predicate.Arguments.get(indexArgument));
        }

        HashSet<Integer> predicateNameIds = this.predicateNameToIds.getOrDefault(predicate.Name, null);
        if (predicateNameIds == null) {
            predicateNameIds = new HashSet<>();
            this.predicateNameToIds.put(predicate.Name, predicateNameIds);
        }

        predicateNameIds.add(this.nextId);

        this.predicateToId.put(predicate, this.nextId);

        if (!Double.isNaN(observedWeight)) {
            this.observedPredicateWeights.put(this.nextId, observedWeight);
        }

        id = this.nextId++;

        return new AbstractMap.SimpleEntry<>(true, id);
    }

    public int getIdForPredicate(ProbabilisticSoftLogicProblem.Predicate predicate) {
        return this.predicateToId.getOrDefault(predicate, -1);
    }

    public Set<Integer> getIdsForPredicateName(String predicateName) {
        return Collections.unmodifiableSet(this.predicateNameToIds.getOrDefault(predicateName, null));
    }

    public void closePredicate(String predicateName) {
        this.closedPredicates.add(predicateName);
    }

    public boolean getIsClosedPredicate(String predicateName) {
        return this.closedPredicates.contains(predicateName);
    }

    public Set<String> getArgumentGroundings(String predicateName, int argumentPosition) {

        List<HashSet<String>> allPredicateGroundings =
            this.argumentGroundings.getOrDefault(predicateName, null);

        if (allPredicateGroundings == null || argumentPosition >= allPredicateGroundings.size()) {
            return Collections.unmodifiableSet(new HashSet<>());
        }

        return Collections.unmodifiableSet(allPredicateGroundings.get(argumentPosition));

    }

    public RandomWalkSampler.Builder<String> addEdgesToRandomWalkSampler(RandomWalkSampler.Builder<String> builder) {

        for (Map.Entry<ProbabilisticSoftLogicProblem.Predicate, Integer> entry : this.predicateToId.entrySet()) {

            int id = entry.getValue();
            double weight = this.getObservedWeight(entry.getValue());
            ProbabilisticSoftLogicProblem.Predicate predicate = entry.getKey();

            if (predicate.Arguments.size() > 2) {
                throw new UnsupportedOperationException("Unable to handle more than 2 arguments for graph based sampling");
            }

            if (Double.isNaN(weight)) {
                builder.addUnobservedEdge(id, predicate.Arguments.get(0), predicate.Arguments.get(1));
            } else {
                builder.addObservedEdge(id, weight, predicate.Arguments.get(0), predicate.Arguments.get(1));
            }

        }

        return builder;

    }

    public int size() {
        return this.predicateToId.size();
    }

    private int nextId = 0;
    private final Set<String> closedPredicates = new HashSet<>();
    private final BiMap<ProbabilisticSoftLogicProblem.Predicate, Integer> predicateToId = HashBiMap.create();
    private final Map<String, HashSet<Integer>> predicateNameToIds = new HashMap<>();
    private final Map<String, ArrayList<HashSet<String>>> argumentGroundings = new HashMap<>();
    private final Map<Integer, Double> observedPredicateWeights = new HashMap<>();
    private final Set<String> entities = new HashSet<>();

}
