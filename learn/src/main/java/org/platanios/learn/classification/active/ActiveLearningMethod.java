package org.platanios.learn.classification.active;

import com.google.common.collect.Iterables;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.statistics.StatisticsUtilities;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum ActiveLearningMethod {
    RANDOM {
        @Override
        public Learning.InstanceToLabel pickInstanceToLabel(
                Learning learning,
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet
        ) {
            List<Learning.InstanceToLabel> instances = pickInstancesToLabel(learning, dataSet, 1);
            return instances.size() > 0 ? instances.get(0) : null;
        }

        @Override
        public List<Learning.InstanceToLabel> pickInstancesToLabel(
                Learning learning,
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet,
                int numberOfInstancesToPick
        ) {
            List<Learning.InstanceToLabel> instances = collectInstances(dataSet);
            if (instances.size() > 0)
                return StatisticsUtilities.sampleWithoutReplacement(
                        instances,
                        Math.min(numberOfInstancesToPick, instances.size())
                );
            else
                return new ArrayList<>();
        }

        private List<Learning.InstanceToLabel> collectInstances(
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet
        ) {
            List<Learning.InstanceToLabel> instances = new ArrayList<>();
            for (Map.Entry<DataInstance<Vector>, Map<Label, Double>> instanceEntry : dataSet.entrySet())
                instances.addAll(instanceEntry
                                         .getValue()
                                         .entrySet()
                                         .stream()
                                         .map(instanceLabelEntry ->
                                                      new Learning.InstanceToLabel(instanceLabelEntry.getKey(),
                                                                                   instanceEntry.getKey()))
                                         .collect(Collectors.toList()));
            return instances;
        }
    },
    UNCERTAINTY_HEURISTIC {
        @Override
        public Learning.InstanceToLabel pickInstanceToLabel(
                Learning learning,
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet
        ) {
            Set<Map.Entry<Learning.InstanceToLabel, Double>> instanceEntropies =
                    computeInstanceEntropies(dataSet).entrySet();
            if (instanceEntropies.size() > 0)
                return instanceEntropies
                        .stream()
                        .max(Map.Entry.comparingByValue(Double::compareTo))
                        .get()
                        .getKey();
            else
                return null;
        }

        @Override
        public List<Learning.InstanceToLabel> pickInstancesToLabel(
                Learning learning,
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet,
                int numberOfInstancesToPick
        ) {
            Set<Map.Entry<Learning.InstanceToLabel, Double>> instanceEntropies =
                    computeInstanceEntropies(dataSet).entrySet();
            if (instanceEntropies.size() > 0)
                return instanceEntropies
                        .stream()
                        .sorted(Collections.reverseOrder(Map.Entry.comparingByValue(Double::compareTo)))
                        .collect(Collectors.toList())
                        .subList(0, Math.min(numberOfInstancesToPick, instanceEntropies.size()))
                        .stream()
                        .map(Map.Entry::getKey)
                        .collect(Collectors.toList());
            else
                return new ArrayList<>();
        }

        private Map<Learning.InstanceToLabel, Double> computeInstanceEntropies(
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet
        ) {
            Map<Learning.InstanceToLabel, Double> instanceEntropies = new HashMap<>();
            for (Map.Entry<DataInstance<Vector>, Map<Label, Double>> instanceEntry : dataSet.entrySet())
                for (Map.Entry<Label, Double> instanceLabelEntry : instanceEntry.getValue().entrySet())
                    instanceEntropies.put(new Learning.InstanceToLabel(instanceLabelEntry.getKey(),
                                                                       instanceEntry.getKey()),
                                          entropy(instanceLabelEntry.getValue()));
            return instanceEntropies;
        }
    },
    CONSTRAINT_PROPAGATION_HEURISTIC {
        @Override
        public Learning.InstanceToLabel pickInstanceToLabel(
                Learning learning,
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet
        ) {
            if (!(learning instanceof ConstrainedLearning))
                throw new IllegalArgumentException("This active learning method can only " +
                                                           "be used with the constrained learner.");
            Set<Map.Entry<Learning.InstanceToLabel, Double>> instanceScores =
                    computeInstanceScores((ConstrainedLearning) learning, dataSet).entrySet();
            if (instanceScores.size() > 0)
                return instanceScores
                        .stream()
                        .max(Map.Entry.comparingByValue(Double::compareTo))
                        .get()
                        .getKey();
            else
                return null;
        }

        @Override
        public List<Learning.InstanceToLabel> pickInstancesToLabel(
                Learning learning,
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet,
                int numberOfInstancesToPick
        ) {
            if (!(learning instanceof ConstrainedLearning))
                throw new IllegalArgumentException("This active learning method can only " +
                                                           "be used with the constrained learner.");
            Set<Map.Entry<Learning.InstanceToLabel, Double>> instanceScores =
                    computeInstanceScores((ConstrainedLearning) learning, dataSet).entrySet();
            if (instanceScores.size() > 0)
                return instanceScores
                        .stream()
                        .sorted(Collections.reverseOrder(Map.Entry.comparingByValue(Double::compareTo)))
                        .collect(Collectors.toList())
                        .subList(0, Math.min(numberOfInstancesToPick, instanceScores.size()))
                        .stream()
                        .map(Map.Entry::getKey)
                        .collect(Collectors.toList());
            else
                return new ArrayList<>();
        }

        private Map<Learning.InstanceToLabel, Double> computeInstanceScores(
                ConstrainedLearning learning,
                Map<DataInstance<Vector>, Map<Label, Double>> dataSet
        ) {
            Map<Learning.InstanceToLabel, Double> instanceScores = new HashMap<>();
            if (learning.getConstraintSet() instanceof ConstraintSet
                    && ((ConstraintSet) learning.getConstraintSet()).getConstraints().size() == 1
                    && Iterables.getOnlyElement(((ConstraintSet) learning.getConstraintSet()).getConstraints()) instanceof MutualExclusionConstraint) {
                for (Map.Entry<DataInstance<Vector>, Map<Label, Double>> instanceEntry : dataSet.entrySet())
                    for (Map.Entry<Label, Double> instanceLabelEntry : instanceEntry.getValue().entrySet())
                        instanceScores.put(new Learning.InstanceToLabel(instanceLabelEntry.getKey(),
                                                                        instanceEntry.getKey()),
                                           instanceLabelEntry.getValue());
                return instanceScores;
            }
//            Map<PredictedDataInstance<V, Double>, Map<Label, Double>> probabilitiesMap = new HashMap<>();
//            for (Map.Entry<Label, DataSet<PredictedDataInstance<V, Double>>> instanceEntry : dataSets.entrySet()) {
//                DataSet<PredictedDataInstance<V, Double>> currentDataSet = instanceEntry.getValue();
//                for (PredictedDataInstance<V, Double> instance : currentDataSet) {
//                    if (!probabilitiesMap.containsKey(instance))
//                        probabilitiesMap.put(instance, new HashMap<>());
//                    if (!probabilitiesMap.get(instance).containsKey(instanceEntry.getKey()))
//                        probabilitiesMap.get(instance).put(instanceEntry.getKey(), instance.label() >= 0.5 ? instance.probability() : 1 - instance.probability());
//                    else
//                        System.out.println("error error error");
//                }
//            }
//            for (Map.Entry<PredictedDataInstance<V, Double>, Map<Label, Double>> instanceEntry : probabilitiesMap.entrySet()) {
//                for (Map.Entry<Label, Double> instanceEntryLabel : instanceEntry.getValue().entrySet()) {
//                    double instanceScore = 0.0;
//                    // Setting label to true and propagating
//                    Map<Label, Boolean> newFixedLabels = new HashMap<>(learning.getFixedLabels(instanceEntry.getKey().name()));
//                    newFixedLabels.put(instanceEntryLabel.getKey(), true);
//                    learning.getConstraintSet().propagate(newFixedLabels);
//                    learning.getFixedLabels(instanceEntry.getKey().name()).keySet().forEach(newFixedLabels::remove);
//                    newFixedLabels.remove(instanceEntryLabel.getKey());
//                    for (Map.Entry<Label, Boolean> labelEntry : newFixedLabels.entrySet()) {
//                        double labelProbability = probabilitiesMap.get(instanceEntry.getKey()).get(labelEntry.getKey());
//                        instanceScore += instanceEntryLabel.getValue() * Math.log(labelEntry.getValue() ? 1 - labelProbability : labelProbability);
//                    }
//                    // Setting label to false and propagating
//                    newFixedLabels = new HashMap<>(learning.getFixedLabels(instanceEntry.getKey().name()));
//                    newFixedLabels.put(instanceEntryLabel.getKey(), false);
//                    learning.getConstraintSet().propagate(newFixedLabels);
//                    learning.getFixedLabels(instanceEntry.getKey().name()).keySet().forEach(newFixedLabels::remove);
//                    newFixedLabels.remove(instanceEntryLabel.getKey());
//                    for (Map.Entry<Label, Boolean> labelEntry : newFixedLabels.entrySet()) {
//                        double labelProbability = probabilitiesMap.get(instanceEntry.getKey()).get(labelEntry.getKey());
//                        instanceScore -= (1 - instanceEntryLabel.getValue()) * Math.log(labelEntry.getValue() ? 1 - labelProbability : labelProbability);
//                    }
//                    instanceScores.put(new Learning.InstanceToLabel<>(instanceEntryLabel.getKey(), instanceEntry.getKey()), instanceScore);
//                }
//            }
            return instanceScores;
        }
    };

    public abstract Learning.InstanceToLabel pickInstanceToLabel(
            Learning learning,
            Map<DataInstance<Vector>, Map<Label, Double>> dataSet
    );

    public abstract List<Learning.InstanceToLabel> pickInstancesToLabel(
            Learning learning,
            Map<DataInstance<Vector>, Map<Label, Double>> dataSet,
            int numberOfInstancesToPick
    );

    protected double entropy(double probability) {
        if (probability > 0)
            return -probability * Math.log(probability) - (1 - probability) * Math.log(1 - probability);
        else
            return 0;
    }
}
