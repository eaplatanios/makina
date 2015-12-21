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
            for (Map.Entry<DataInstance<Vector>, Map<Label, Double>> instanceEntry : dataSet.entrySet()) {
                for (Map.Entry<Label, Double> instanceEntryLabel : instanceEntry.getValue().entrySet()) {
                    List<Double> instanceScoreTerms = new ArrayList<>();
                    // Setting label to true and propagating
                    Map<Label, Boolean> fixedLabels = new HashMap<>(learning.getLabels(instanceEntry.getKey()));
                    fixedLabels.put(instanceEntryLabel.getKey(), true);
                    learning.getConstraintSet().propagate(fixedLabels);
                    learning.getLabels(instanceEntry.getKey()).keySet().forEach(fixedLabels::remove);
                    fixedLabels.remove(instanceEntryLabel.getKey());
//                    instanceScoreTerms.add(Math.log(instanceEntryLabel.getValue()));
//                    instanceScoreTerms.add(instanceEntryLabel.getValue());
                    double term = 0.0;
                    for (Map.Entry<Label, Boolean> labelEntry : fixedLabels.entrySet()) {
                        double labelProbability = instanceEntry.getValue().get(labelEntry.getKey());
                        term += instanceEntryLabel.getValue() *
                                (labelEntry.getValue() ? 1 - labelProbability : labelProbability);
                    }
                    instanceScoreTerms.add(term);
//                    if (term > 0.0)
//                        instanceScoreTerms.add(Math.log(instanceEntryLabel.getValue()) + Math.log(term));
                    // Setting label to false and propagating
                    fixedLabels = new HashMap<>(learning.getLabels(instanceEntry.getKey()));
                    fixedLabels.put(instanceEntryLabel.getKey(), false);
                    learning.getConstraintSet().propagate(fixedLabels);
                    learning.getLabels(instanceEntry.getKey()).keySet().forEach(fixedLabels::remove);
                    fixedLabels.remove(instanceEntryLabel.getKey());
                    term = 0.0;
                    for (Map.Entry<Label, Boolean> labelEntry : fixedLabels.entrySet()) {
                        double labelProbability = instanceEntry.getValue().get(labelEntry.getKey());
                        term += labelEntry.getValue() ? 1 - labelProbability : labelProbability;
                    }
//                    if (term > 0.0)
//                        instanceScoreTerms.add(Math.log(1 - instanceEntryLabel.getValue()) + Math.log(term));
//                    if (instanceScoreTerms.size() > 0)
                        instanceScores.put(
                                new Learning.InstanceToLabel(instanceEntryLabel.getKey(), instanceEntry.getKey()),
                                instanceScoreTerms.stream().mapToDouble(number -> number).sum()
                        );
//                    else
//                        instanceScores.put(
//                                new Learning.InstanceToLabel(instanceEntryLabel.getKey(), instanceEntry.getKey()),
//                                Double.NEGATIVE_INFINITY
//                        );
//                    instanceScores.put(
//                            new Learning.InstanceToLabel(instanceEntryLabel.getKey(), instanceEntry.getKey()),
//                            Math.log(instanceEntryLabel.getValue())
//                    );
                }
            }
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
