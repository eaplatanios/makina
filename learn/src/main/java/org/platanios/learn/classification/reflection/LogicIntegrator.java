package org.platanios.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;
import org.platanios.learn.classification.constraint.SubsumptionConstraint;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.logic.InMemoryLogicManager;
import org.platanios.learn.logic.LogicManager;
import org.platanios.learn.logic.LukasiewiczLogic;
import org.platanios.learn.logic.ProbabilisticSoftLogic;
import org.platanios.learn.logic.formula.*;
import org.platanios.learn.logic.grounding.GroundPredicate;
import org.platanios.learn.math.matrix.Vector;

import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogicIntegrator {
    private static final Logger logger = LogManager.getLogger("Classification / Logic Integrator");

    private final LogicManager logicManager;
    private final Set<Label> labels;
    private final Map<Label, Set<Integer>> labelClassifiers;
    private final Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet;
    private final Set<Constraint> constraints;
    private final boolean estimateErrorRates;
    private final BiMap<Long, DataInstance<Vector>> instanceKeysMap;
    private final BiMap<Long, Label> labelKeysMap;
    private final BiMap<Long, Integer> classifierKeysMap;
    private final EntityType instanceType;
    private final EntityType labelType;
    private final EntityType classifierType;
    private final Predicate labelPredicate;
    private final Predicate equalLabelsPredicate;
    private final Predicate labelPredictionPredicate;
    private final Predicate errorRatePredicate;

    private final ProbabilisticSoftLogic.Builder pslBuilder;

    public static class Builder {
        private final Set<Label> labels;
        private final Map<Label, Set<Integer>> labelClassifiers;
        private final Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet;

        private LogicManager logicManager = new InMemoryLogicManager(new LukasiewiczLogic());
        private Set<Constraint> constraints = new HashSet<>();
        private boolean estimateErrorRates = false;

        public Builder(Map<Label, Set<Integer>> labelClassifiers,
                       Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet) {
            this.labels = labelClassifiers.keySet();
            this.labelClassifiers = labelClassifiers;
            this.dataSet = dataSet;
        }

        public Builder logicManager(LogicManager logicManager) {
            this.logicManager = logicManager;
            return this;
        }

        public Builder addConstraint(MutualExclusionConstraint constraint) {
            constraints.add(constraint);
            return this;
        }

        public Builder addConstraint(SubsumptionConstraint constraint) {
            constraints.add(constraint);
            return this;
        }

        public LogicIntegrator build() {
            return new LogicIntegrator(this);
        }
    }

    private LogicIntegrator(Builder builder) {
        logicManager = builder.logicManager;
        labels = builder.labels;
        labelClassifiers = builder.labelClassifiers;
        dataSet = builder.dataSet;
        constraints = builder.constraints;
        estimateErrorRates = builder.estimateErrorRates;
        instanceKeysMap = HashBiMap.create(dataSet.size());
        labelKeysMap = HashBiMap.create(labels.size());
        classifierKeysMap = HashBiMap.create(
                (int) labelClassifiers.values().stream().flatMap(Collection::stream).count()
        );
        pslBuilder = new ProbabilisticSoftLogic.Builder(logicManager);
        final long[] currentInstanceKey = { 0 };
        final long[] currentLabelKey = { 0 };
        final long[] currentClassifierKey = { 0 };
        dataSet.keySet().forEach(instance -> {
            if (!instanceKeysMap.containsValue(instance))
                instanceKeysMap.put(currentInstanceKey[0]++, instance);
        });
        labels.forEach(label -> {
            if (!labelKeysMap.containsValue(label))
                labelKeysMap.put(currentLabelKey[0]++, label);
            labelClassifiers.get(label)
                    .stream()
                    .filter(classifierId -> !classifierKeysMap.containsValue(classifierId))
                    .forEach(classifierId -> classifierKeysMap.put(currentClassifierKey[0]++, classifierId));
        });
        logger.info("Adding entity types to the logic manager.");
        instanceType = logicManager.addEntityType("{instance}", instanceKeysMap.keySet());
        labelType = logicManager.addEntityType("{label}", labelKeysMap.keySet());
        classifierType = logicManager.addEntityType("{classifier}", classifierKeysMap.keySet());
        logger.info("Adding predicates to the logic manager.");
        List<EntityType> argumentTypes = new ArrayList<>(2);
        argumentTypes.add(instanceType);
        argumentTypes.add(labelType);
        labelPredicate = logicManager.addPredicate("LABEL", argumentTypes, false);
        argumentTypes = new ArrayList<>(2);
        argumentTypes.add(labelType);
        argumentTypes.add(labelType);
        equalLabelsPredicate = logicManager.addPredicate("EQUAL_LABELS", argumentTypes, true);
        argumentTypes = new ArrayList<>(3);
        argumentTypes.add(instanceType);
        argumentTypes.add(classifierType);
        argumentTypes.add(labelType);
        labelPredictionPredicate = logicManager.addPredicate("LABEL_PREDICTION", argumentTypes, false);
        argumentTypes = new ArrayList<>(2);
        argumentTypes.add(classifierType);
        argumentTypes.add(labelType);
        errorRatePredicate = logicManager.addPredicate("ERROR_RATE", argumentTypes, false);
        logger.info("Adding rules to the probabilistic soft logic builder.");
        Variable instanceVariable = new Variable(0, "I", instanceType);
        Variable classifierVariable = new Variable(1, "C", classifierType);
        Variable label1Variable = new Variable(2, "L1", labelType);
        Variable label2Variable = new Variable(3, "L2", labelType);
        // Mutual Exclusion Rule
        double power = 1;
        double weight = 1;
        List<Formula> bodyFormulas = new ArrayList<>();
        bodyFormulas.add(new Atom(labelPredictionPredicate, Arrays.asList(instanceVariable,
                                                                          classifierVariable,
                                                                          label1Variable)));
        bodyFormulas.add(new Atom(labelPredicate, Arrays.asList(instanceVariable, label2Variable)));
        bodyFormulas.add(new Negation(new Atom(equalLabelsPredicate, Arrays.asList(label1Variable,
                                                                                   label2Variable))));
        List<Formula> headFormulas = new ArrayList<>();
        headFormulas.add(new Atom(errorRatePredicate, Arrays.asList(classifierVariable, label1Variable)));
        pslBuilder.addLogicRule(new ProbabilisticSoftLogic.LogicRule(bodyFormulas,
                                                                     headFormulas,
                                                                     power,
                                                                     weight));
        // Ensemble Classifier Rule
        power = 1;
        weight = 1;
        bodyFormulas = new ArrayList<>();
        bodyFormulas.add(new Atom(labelPredictionPredicate, Arrays.asList(instanceVariable,
                                                                          classifierVariable,
                                                                          label1Variable)));
        bodyFormulas.add(new Negation(new Atom(errorRatePredicate, Arrays.asList(classifierVariable,
                                                                                 label1Variable))));
        headFormulas = new ArrayList<>();
        headFormulas.add(new Atom(labelPredicate, Arrays.asList(instanceVariable, label1Variable)));
        pslBuilder.addLogicRule(new ProbabilisticSoftLogic.LogicRule(bodyFormulas,
                                                                     headFormulas,
                                                                     power,
                                                                     weight));
        logger.info("Adding ground predicates and rules to the probabilistic soft logic builder.");
        labels.forEach(label -> logicManager.addGroundPredicate(equalLabelsPredicate,
                                                                Arrays.asList(labelKeysMap.inverse().get(label),
                                                                              labelKeysMap.inverse().get(label)),
                                                                1.0));
        for (Map.Entry<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSetEntry : dataSet.entrySet()) {
            for (Map.Entry<Label, Map<Integer, Double>> dataInstanceEntry : dataSetEntry.getValue().entrySet()) {
                for (Map.Entry<Integer, Double> classifierEntry : dataInstanceEntry.getValue().entrySet()) {
                    List<Long> assignment = new ArrayList<>(3);
                    assignment.add(instanceKeysMap.inverse().get(dataSetEntry.getKey()));
                    assignment.add(classifierKeysMap.inverse().get(classifierEntry.getKey()));
                    assignment.add(labelKeysMap.inverse().get(dataInstanceEntry.getKey()));
                    logicManager.addGroundPredicate(labelPredictionPredicate, assignment, classifierEntry.getValue());
//                    List<GroundPredicate> rule1Predicates = new ArrayList<>(4);
//                    List<GroundPredicate> rule2Predicates = new ArrayList<>(3);
//                    List<Long> assignment = new ArrayList<>(3);
//                    assignment.add(instanceKeysMap.inverse().get(dataSetEntry.getKey()));
//                    assignment.add(classifierKeysMap.inverse().get(classifierEntry.getKey()));
//                    assignment.add(labelKeysMap.inverse().get(dataInstanceEntry.getKey()));
//                    rule1Predicates.add(logicManager.addGroundPredicate(labelPredictionPredicate,
//                                                                        assignment,
//                                                                        classifierEntry.getValue()));
//                    rule2Predicates.add(logicManager.addGroundPredicate(labelPredictionPredicate,
//                                                                        assignment,
//                                                                        classifierEntry.getValue()));
//                    assignment = new ArrayList<>(2);
//                    assignment.add(instanceKeysMap.inverse().get(dataSetEntry.getKey()));
//                    assignment.add(labelKeysMap.inverse().get(dataInstanceEntry.getKey()));
//                    rule1Predicates.add(logicManager.addGroundPredicate(labelPredicate, assignment, null));
//                    rule2Predicates.add(logicManager.addGroundPredicate(labelPredicate, assignment, null));
//                    assignment = new ArrayList<>(2);
//                    assignment.add(labelKeysMap.inverse().get(dataInstanceEntry.getKey()));
//                    assignment.add(labelKeysMap.inverse().get(dataInstanceEntry.getKey()));
//                    rule1Predicates.add(logicManager.addGroundPredicate(equalLabelsPredicate, assignment, 1.0));
//                    assignment = new ArrayList<>(2);
//                    assignment.add(classifierKeysMap.inverse().get(classifierEntry.getKey()));
//                    assignment.add(labelKeysMap.inverse().get(dataInstanceEntry.getKey()));
//                    rule1Predicates.add(logicManager.addGroundPredicate(errorRatePredicate, assignment, null));
//                    rule2Predicates.add(logicManager.addGroundPredicate(errorRatePredicate, assignment, null));
//                    pslBuilder.addRule(rule1Predicates, new boolean[] { true, true, false, false }, 1, 1);
//                    pslBuilder.addRule(rule2Predicates, new boolean[] { true, false, false }, 1, 1);
                }
            }
        }
    }

    public Output integratePredictions() {
        logger.info("Starting inference...");
        ProbabilisticSoftLogic psl = pslBuilder.build();
        List<GroundPredicate> inferredGroundPredicates = psl.solve();
        Map<DataInstance<Vector>, Map<Label, Double>> integratedDataSet = new HashMap<>();
        Map<Label, Map<Integer, Double>> errorRates = new HashMap<>();
        for (DataInstance<Vector> instance : dataSet.keySet())
            integratedDataSet.put(instance, new HashMap<>());
        for (Label label : labels)
            errorRates.put(label, new HashMap<>());
        inferredGroundPredicates
                .stream()
                .forEach(inferredGroundPredicate -> {
                    if (inferredGroundPredicate.getPredicate().equals(labelPredicate)) {
                        DataInstance<Vector> instance = instanceKeysMap.get(inferredGroundPredicate.getArguments().get(0));
                        Label label = labelKeysMap.get(inferredGroundPredicate.getArguments().get(1));
                        integratedDataSet.get(instance).put(label, inferredGroundPredicate.getValue());
                    } else if (inferredGroundPredicate.getPredicate().equals(errorRatePredicate)) {
                        int classifierId = classifierKeysMap.get(inferredGroundPredicate.getArguments().get(0));
                        Label label = labelKeysMap.get(inferredGroundPredicate.getArguments().get(1));
                        errorRates.get(label).put(classifierId, inferredGroundPredicate.getValue());
                    }
                });
        return new Output(integratedDataSet, errorRates);
    }

    public static class Output {
        private final Map<DataInstance<Vector>, Map<Label, Double>> integratedDataSet;
        private final Map<Label, Map<Integer, Double>> errorRates;

        public Output(Map<DataInstance<Vector>, Map<Label, Double>> integratedDataSet,
                      Map<Label, Map<Integer, Double>> errorRates) {
            this.integratedDataSet = integratedDataSet;
            this.errorRates = errorRates;
        }

        public Map<DataInstance<Vector>, Map<Label, Double>> getIntegratedDataSet() {
            return integratedDataSet;
        }

        public Map<Label, Map<Integer, Double>> getErrorRates() {
            return errorRates;
        }
    }
}