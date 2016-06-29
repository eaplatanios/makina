package makina.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import makina.learn.classification.constraint.Constraint;
import makina.learn.classification.constraint.MutualExclusionConstraint;
import makina.learn.classification.constraint.SubsumptionConstraint;
import makina.learn.logic.ProbabilisticSoftLogic;
import makina.learn.logic.formula.*;
import makina.learn.logic.grounding.GroundPredicate;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.learn.classification.Label;
import makina.learn.logic.InMemoryLogicManager;
import makina.learn.logic.LogicManager;
import makina.learn.logic.LukasiewiczLogic;

import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class LogicIntegrator extends Integrator {
    private static final Logger logger = LogManager.getLogger("Classification / Logic Integrator");

    private final LogicManager logicManager;
    private final Set<Label> labels;
    private final Set<Integer> classifiers;
    private final Set<Constraint> constraints;
    private final boolean sampleErrorRatesEstimates;
    private final boolean logProgress;
    private final ProbabilisticSoftLogic psl;
    private final BiMap<Long, Integer> instanceKeysMap;
    private final BiMap<Long, Label> labelKeysMap;
    private final BiMap<Long, Integer> classifierKeysMap;
    private final EntityType instanceType;
    private final EntityType labelType;
    private final EntityType classifierType;
    private final Predicate mutualExclusionPredicate;
    private final Predicate subsumptionPredicate;
    private final Predicate labelPredicate;
    private final Predicate equalLabelsPredicate;
    private final Predicate labelPredictionPredicate;
    private final Predicate errorRatePredicate;

    private boolean needsInference = true;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends Integrator.AbstractBuilder<T> {
        private final Set<Label> labels = new HashSet<>();
        private final Set<Integer> classifiers = new HashSet<>();

        private final Integrator.Data<Integrator.Data.ObservedInstance> observedData;

        private LogicManager logicManager = new InMemoryLogicManager(new LukasiewiczLogic());
        private Set<Constraint> constraints = new HashSet<>();
        private boolean sampleErrorRatesEstimates = false;
        private boolean logProgress = false;

        public AbstractBuilder(Integrator.Data<Integrator.Data.PredictedInstance> predictedData) {
            this(predictedData, null);
        }

        public AbstractBuilder(Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                               Integrator.Data<Integrator.Data.ObservedInstance> observedData) {
            super(predictedData);
            if (observedData != null)
                extractLabelsSet(observedData);
            extractLabelsSet(predictedData);
            extractClassifiersSet(predictedData);
            this.observedData = observedData;
        }

        private void extractLabelsSet(Integrator.Data<?> data) {
            data.stream().map(Integrator.Data.Instance::label).forEach(labels::add);
        }

        private void extractClassifiersSet(Integrator.Data<Integrator.Data.PredictedInstance> predictedData) {
            predictedData.stream()
                    .map(Integrator.Data.PredictedInstance::functionId)
                    .forEach(classifiers::add);
        }

        public T logicManager(LogicManager logicManager) {
            this.logicManager = logicManager;
            return self();
        }

        public T addConstraint(MutualExclusionConstraint constraint) {
            constraints.add(constraint);
            return self();
        }

        public T addConstraint(SubsumptionConstraint constraint) {
            constraints.add(constraint);
            return self();
        }

        public T addConstraints(Set<Constraint> constraints) {
            this.constraints.addAll(constraints);
            return self();
        }

        public T sampleErrorRatesEstimates(boolean sampleErrorRatesEstimates) {
            this.sampleErrorRatesEstimates = sampleErrorRatesEstimates;
            return self();
        }

        public T logProgress(boolean logProgress) {
            this.logProgress = logProgress;
            return self();
        }

        public LogicIntegrator build() {
            return new LogicIntegrator(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(Integrator.Data<Integrator.Data.PredictedInstance> predictedData) {
            super(predictedData);
        }

        public Builder(Integrator.Data<Integrator.Data.PredictedInstance> predictedData,
                       Integrator.Data<Integrator.Data.ObservedInstance> observedData) {
            super(predictedData, observedData);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private LogicIntegrator(AbstractBuilder<?> builder) {
        super(builder);
        logicManager = builder.logicManager;
        labels = builder.labels;
        classifiers = builder.classifiers;
        constraints = builder.constraints;
        sampleErrorRatesEstimates = builder.sampleErrorRatesEstimates;
        logProgress = builder.logProgress;
        instanceKeysMap = HashBiMap.create(data.size());
        labelKeysMap = HashBiMap.create(labels.size());
        classifierKeysMap = HashBiMap.create((int) classifiers.stream().count());
        ProbabilisticSoftLogic.Builder pslBuilder = new ProbabilisticSoftLogic.Builder(logicManager);
        final long[] currentInstanceKey = {0};
        final long[] currentLabelKey = {0};
        final long[] currentClassifierKey = {0};
        if (builder.observedData != null)
            builder.observedData.stream().map(Integrator.Data.Instance::id).forEach(instance -> {
                if (!instanceKeysMap.containsValue(instance))
                    instanceKeysMap.put(currentInstanceKey[0]++, instance);
            });
        data.stream().map(Integrator.Data.Instance::id).forEach(instance -> {
            if (!instanceKeysMap.containsValue(instance))
                instanceKeysMap.put(currentInstanceKey[0]++, instance);
        });
        labels.forEach(label -> labelKeysMap.put(currentLabelKey[0]++, label));
        classifiers.forEach(classifier -> classifierKeysMap.put(currentClassifierKey[0]++, classifier));
        if (logProgress)
            logger.info("Adding entity types to the logic manager.");
        instanceType = logicManager.addEntityType("{instance}", instanceKeysMap.keySet());
        labelType = logicManager.addEntityType("{label}", labelKeysMap.keySet());
        classifierType = logicManager.addEntityType("{classifier}", classifierKeysMap.keySet());
        if (logProgress)
            logger.info("Adding predicates to the logic manager.");
        List<EntityType> argumentTypes = new ArrayList<>(2);
        argumentTypes.add(labelType);
        argumentTypes.add(labelType);
        mutualExclusionPredicate = logicManager.addPredicate("MUTUAL_EXCLUSION", argumentTypes, true);
        subsumptionPredicate = logicManager.addPredicate("SUBSUMPTION", argumentTypes, true);
        argumentTypes = new ArrayList<>(2);
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
        if (logProgress)
            logger.info("Adding rules to the probabilistic soft logic builder.");
        Variable instanceVariable = new Variable(0, "I", instanceType);
        Variable classifierVariable = new Variable(1, "C", classifierType);
        Variable label1Variable = new Variable(2, "L1", labelType);
        Variable label2Variable = new Variable(3, "L2", labelType);
        // Mutual Exclusion Rule
        double power = 2;
        double weight = 1;
        List<Formula> bodyFormulas = new ArrayList<>();
        bodyFormulas.add(new Atom(mutualExclusionPredicate, Arrays.asList(label1Variable, label2Variable)));
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
        // Subsumption Rule
        power = 2;
        weight = 1;
        bodyFormulas = new ArrayList<>();
        bodyFormulas.add(new Atom(subsumptionPredicate, Arrays.asList(label1Variable, label2Variable)));
        bodyFormulas.add(new Negation(new Atom(labelPredictionPredicate, Arrays.asList(instanceVariable,
                                                                                       classifierVariable,
                                                                                       label1Variable))));
        bodyFormulas.add(new Atom(labelPredicate, Arrays.asList(instanceVariable, label2Variable)));
        bodyFormulas.add(new Negation(new Atom(equalLabelsPredicate, Arrays.asList(label1Variable,
                                                                                   label2Variable))));
        headFormulas = new ArrayList<>();
        headFormulas.add(new Atom(errorRatePredicate, Arrays.asList(classifierVariable, label1Variable)));
        pslBuilder.addLogicRule(new ProbabilisticSoftLogic.LogicRule(bodyFormulas,
                                                                     headFormulas,
                                                                     power,
                                                                     weight));
        // Ensemble Classifier Rule #1
        power = 2;
        weight = 1;
        bodyFormulas = new ArrayList<>();
        bodyFormulas.add(new Atom(labelPredictionPredicate, Arrays.asList(instanceVariable,
                                                                          classifierVariable,
                                                                          label1Variable)));
        bodyFormulas.add(new Negation(new Atom(errorRatePredicate, Arrays.asList(classifierVariable, label1Variable))));
        headFormulas = new ArrayList<>();
        headFormulas.add(new Atom(labelPredicate, Arrays.asList(instanceVariable, label1Variable)));
        pslBuilder.addLogicRule(new ProbabilisticSoftLogic.LogicRule(bodyFormulas,
                                                                     headFormulas,
                                                                     power,
                                                                     weight));
        // Ensemble Classifier Rule #2
        power = 2;
        weight = 1;
        bodyFormulas = new ArrayList<>();
        bodyFormulas.add(new Atom(labelPredictionPredicate, Arrays.asList(instanceVariable,
                                                                          classifierVariable,
                                                                          label1Variable)));
        bodyFormulas.add(new Atom(errorRatePredicate, Arrays.asList(classifierVariable, label1Variable)));
        headFormulas = new ArrayList<>();
        headFormulas.add(new Negation(new Atom(labelPredicate, Arrays.asList(instanceVariable, label1Variable))));
        pslBuilder.addLogicRule(new ProbabilisticSoftLogic.LogicRule(bodyFormulas,
                                                                     headFormulas,
                                                                     power,
                                                                     weight));
        // Ensemble Classifier Rule #3
        power = 2;
        weight = 1;
        bodyFormulas = new ArrayList<>();
        bodyFormulas.add(new Negation(new Atom(labelPredictionPredicate, Arrays.asList(instanceVariable,
                                                                                       classifierVariable,
                                                                                       label1Variable))));
        bodyFormulas.add(new Negation(new Atom(errorRatePredicate, Arrays.asList(classifierVariable,
                                                                                 label1Variable))));
        headFormulas = new ArrayList<>();
        headFormulas.add(new Negation(new Atom(labelPredicate, Arrays.asList(instanceVariable, label1Variable))));
        pslBuilder.addLogicRule(new ProbabilisticSoftLogic.LogicRule(bodyFormulas,
                                                                     headFormulas,
                                                                     power,
                                                                     weight));
        // Ensemble Classifier Rule #4
        power = 2;
        weight = 1;
        bodyFormulas = new ArrayList<>();
        bodyFormulas.add(new Negation(new Atom(labelPredictionPredicate, Arrays.asList(instanceVariable,
                                                                                       classifierVariable,
                                                                                       label1Variable))));
        bodyFormulas.add(new Atom(errorRatePredicate, Arrays.asList(classifierVariable, label1Variable)));
        headFormulas = new ArrayList<>();
        headFormulas.add(new Atom(labelPredicate, Arrays.asList(instanceVariable, label1Variable)));
        pslBuilder.addLogicRule(new ProbabilisticSoftLogic.LogicRule(bodyFormulas,
                                                                     headFormulas,
                                                                     power,
                                                                     weight));
        if (logProgress)
            logger.info("Adding ground predicates and rules to the probabilistic soft logic builder.");
        labels.forEach(label -> logicManager.addGroundPredicate(equalLabelsPredicate,
                                                                Arrays.asList(labelKeysMap.inverse().get(label),
                                                                              labelKeysMap.inverse().get(label)),
                                                                1.0));
        for (Constraint constraint : constraints) {
            if (constraint instanceof MutualExclusionConstraint) {
                List<Label> labelsList = new ArrayList<>(((MutualExclusionConstraint) constraint).getLabels());
                for (int label1Index = 0; label1Index < labelsList.size(); label1Index++)
                    for (int label2Index = 0; label2Index < labelsList.size(); label2Index++)
                        if (label1Index != label2Index)
                            logicManager.addGroundPredicate(
                                    mutualExclusionPredicate,
                                    Arrays.asList(labelKeysMap.inverse().get(labelsList.get(label1Index)),
                                                  labelKeysMap.inverse().get(labelsList.get(label2Index))),
                                    1.0
                            );
            } else if (constraint instanceof SubsumptionConstraint) {
                Label parentLabel = ((SubsumptionConstraint) constraint).getParentLabel();
                Label childLabel = ((SubsumptionConstraint) constraint).getChildLabel();
                logicManager.addGroundPredicate(
                        subsumptionPredicate,
                        Arrays.asList(labelKeysMap.inverse().get(parentLabel),
                                      labelKeysMap.inverse().get(childLabel)),
                        1.0
                );
            }
        }
        if (builder.observedData != null)
            for (Integrator.Data.ObservedInstance instance : builder.observedData)
                logicManager.addOrReplaceGroundPredicate(labelPredicate,
                                                         Arrays.asList(
                                                                 instanceKeysMap.inverse().get(instance.id()),
                                                                 labelKeysMap.inverse().get(instance.label())
                                                         ),
                                                         instance.value() ? 1.0 : 0.0);
        if (logProgress) {
            logger.info("Number of positive instances: " + data.stream().filter(i -> i.value() >= 0.5).count());
            logger.info("Number of negative instances: " + data.stream().filter(i -> i.value() < 0.5).count());
        }
        for (Integrator.Data.PredictedInstance instance : data) {
//            List<Long> assignment = new ArrayList<>(3);
//            assignment.add(instanceKeysMap.inverse().get(instance.id()));
//            assignment.add(classifierKeysMap.inverse().get(instance.functionId()));
//            assignment.add(labelKeysMap.inverse().get(instance.label()));
//            if (logicManager.checkIfGroundPredicateExists(labelPredictionPredicate, assignment))
//                continue;
//            logicManager.addGroundPredicate(labelPredictionPredicate, assignment, instance.value());
            List<GroundPredicate> mutualExclusionRulePredicates = new ArrayList<>(4);
            List<GroundPredicate> subsumptionRulePredicates = new ArrayList<>(4);
            List<GroundPredicate> ensembleRulePredicates = new ArrayList<>(3);
            List<GroundPredicate> majorityVotePriorRulePredicates = new ArrayList<>(2);
            List<Long> assignment = new ArrayList<>(3);
            assignment.add(instanceKeysMap.inverse().get(instance.id()));
            assignment.add(classifierKeysMap.inverse().get(instance.functionId()));
            assignment.add(labelKeysMap.inverse().get(instance.label()));
            if (logicManager.checkIfGroundPredicateExists(labelPredictionPredicate, assignment))
                continue;
            GroundPredicate predictionPredicate = logicManager.addGroundPredicate(labelPredictionPredicate,
                                                                                  assignment,
                                                                                  instance.value());
            mutualExclusionRulePredicates.add(predictionPredicate);
            subsumptionRulePredicates.add(predictionPredicate);
            ensembleRulePredicates.add(predictionPredicate);
            majorityVotePriorRulePredicates.add(predictionPredicate);
            assignment = new ArrayList<>(2);
            assignment.add(classifierKeysMap.inverse().get(instance.functionId()));
            assignment.add(labelKeysMap.inverse().get(instance.label()));
            GroundPredicate errorPredicate = logicManager.addGroundPredicate(errorRatePredicate, assignment, null);
            mutualExclusionRulePredicates.add(errorPredicate);
            subsumptionRulePredicates.add(errorPredicate);
            ensembleRulePredicates.add(errorPredicate);
            assignment = new ArrayList<>(2);
            assignment.add(instanceKeysMap.inverse().get(instance.id()));
            assignment.add(labelKeysMap.inverse().get(instance.label()));
            GroundPredicate integratedLabelPredicate = logicManager.addGroundPredicate(labelPredicate, assignment, null);
            ensembleRulePredicates.add(integratedLabelPredicate);
            majorityVotePriorRulePredicates.add(integratedLabelPredicate);
            pslBuilder.addRule(ensembleRulePredicates, new boolean[] { true, false, false }, 2, 1);
            pslBuilder.addRule(ensembleRulePredicates, new boolean[] { true, true, true }, 2, 1);
            pslBuilder.addRule(ensembleRulePredicates, new boolean[] { false, false, true }, 2, 1);
            pslBuilder.addRule(ensembleRulePredicates, new boolean[] { false, true, false }, 2, 1);
            pslBuilder.addRule(majorityVotePriorRulePredicates, new boolean[] { true, false }, 1, 1);
            pslBuilder.addRule(majorityVotePriorRulePredicates, new boolean[] { false, true }, 1, 1);
            for (Constraint constraint : constraints) {
                if (constraint instanceof MutualExclusionConstraint) {
                    List<Label> labelsList = new ArrayList<>(((MutualExclusionConstraint) constraint).getLabels());
                    if (labelsList.contains(instance.label())) {
                        for (Label label : labelsList.stream().filter(l -> !l.equals(instance.label())).collect(Collectors.toList())) {
                            List<GroundPredicate> mutualExclusionRulePredicatesCopy = new ArrayList<>(mutualExclusionRulePredicates);
                            assignment = new ArrayList<>(2);
                            assignment.add(instanceKeysMap.inverse().get(instance.id()));
                            assignment.add(labelKeysMap.inverse().get(label));
                            if (logicManager.checkIfGroundPredicateExists(labelPredicate, assignment))
                                mutualExclusionRulePredicatesCopy.add(logicManager.getGroundPredicate(labelPredicate, assignment));
                            else
                                mutualExclusionRulePredicatesCopy.add(logicManager.addGroundPredicate(labelPredicate, assignment, null));
                            assignment = new ArrayList<>(2);
                            assignment.add(labelKeysMap.inverse().get(instance.label()));
                            assignment.add(labelKeysMap.inverse().get(label));
                            mutualExclusionRulePredicatesCopy.add(logicManager.getGroundPredicate(mutualExclusionPredicate, assignment));
                            pslBuilder.addRule(mutualExclusionRulePredicatesCopy, new boolean[] { true, false, true, true }, 2, 1);
                        }
                    }
                } else if (constraint instanceof SubsumptionConstraint) {
                    if (((SubsumptionConstraint) constraint).getParentLabel().equals(instance.label())) {
                        Label childLabel = ((SubsumptionConstraint) constraint).getChildLabel();
                        List<GroundPredicate> subsumptionRulePredicatesCopy = new ArrayList<>(subsumptionRulePredicates);
                        assignment = new ArrayList<>(2);
                        assignment.add(instanceKeysMap.inverse().get(instance.id()));
                        assignment.add(labelKeysMap.inverse().get(childLabel));
                        if (logicManager.checkIfGroundPredicateExists(labelPredicate, assignment))
                            subsumptionRulePredicatesCopy.add(logicManager.getGroundPredicate(labelPredicate, assignment));
                        else
                            subsumptionRulePredicatesCopy.add(logicManager.addGroundPredicate(labelPredicate, assignment, null));
                        assignment = new ArrayList<>(2);
                        assignment.add(labelKeysMap.inverse().get(instance.label()));
                        assignment.add(labelKeysMap.inverse().get(childLabel));
                        subsumptionRulePredicatesCopy.add(logicManager.getGroundPredicate(subsumptionPredicate, assignment));
                        pslBuilder.addRule(subsumptionRulePredicatesCopy, new boolean[] { false, false, true, true }, 2, 1);
                    }
                }
            }
        }
        psl = pslBuilder.build(false);
    }

    // TODO: Assumes that the corresponding data instance and label are included in the un-fixed data set.
    public void fixDataInstanceLabel(Integer instanceID, Label label, boolean value) {
        List<Long> assignment = new ArrayList<>(2);
        assignment.add(instanceKeysMap.inverse().get(instanceID));
        assignment.add(labelKeysMap.inverse().get(label));
        psl.fixDataInstanceLabel(logicManager.addOrReplaceGroundPredicate(labelPredicate, assignment, value ? 1.0 : 0.0));
        data.stream()
                .filter(instance -> instance.id() == instanceID && instance.label() == label)
                .forEach(instance -> {
                    List<Long> instanceAssignment = new ArrayList<>(3);
                    instanceAssignment.add(instanceKeysMap.inverse().get(instanceID));
                    instanceAssignment.add(classifierKeysMap.inverse().get(instance.functionId()));
                    instanceAssignment.add(labelKeysMap.inverse().get(label));
                    logicManager.removeGroundPredicate(labelPredictionPredicate, instanceAssignment);
                });
        needsInference = true;
    }

    @Override
    public ErrorRates errorRates() {
        performInference();
        return errorRates;
    }

    @Override
    public Integrator.Data<Data.PredictedInstance> integratedData() {
        performInference();
        return integratedData;
    }

    private void performInference() {
        if (!needsInference)
            return;
        if (logProgress)
            logger.info("Starting inference...");
        List<GroundPredicate> inferredGroundPredicates = psl.solve();
        List<Integrator.Data.PredictedInstance> integratedInstances = new ArrayList<>();
        List<ErrorRates.Instance> errorRatesInstances = new ArrayList<>();
        Map<Label, Map<Integer, Boolean>> integratedPredictions = new HashMap<>();
        if (sampleErrorRatesEstimates)
            for (Label label : labels)
                integratedPredictions.put(label, new HashMap<>());
        inferredGroundPredicates
                .stream()
                .forEach(inferredGroundPredicate -> {
                    if (inferredGroundPredicate.getPredicate().equals(labelPredicate)) {
                        Integer instanceID = instanceKeysMap.get(inferredGroundPredicate.getArguments().get(0));
                        Label label = labelKeysMap.get(inferredGroundPredicate.getArguments().get(1));
                        integratedInstances.add(new Integrator.Data.PredictedInstance(
                                instanceID, label, -1, inferredGroundPredicate.getValue())
                        );
                        if (sampleErrorRatesEstimates)
                            integratedPredictions.get(label).put(instanceID, inferredGroundPredicate.getValue() >= 0.5);
                    } else if (!sampleErrorRatesEstimates
                            && inferredGroundPredicate.getPredicate().equals(errorRatePredicate)) {
                        int classifierId = classifierKeysMap.get(inferredGroundPredicate.getArguments().get(0));
                        Label label = labelKeysMap.get(inferredGroundPredicate.getArguments().get(1));
                        errorRatesInstances.add(new ErrorRates.Instance(
                                label, classifierId, inferredGroundPredicate.getValue())
                        );
                    }
                });
        if (sampleErrorRatesEstimates)
            for (Label label : labels)
                data.stream()
                        .filter(i -> i.label().equals(label))
                        .map(Data.PredictedInstance::functionId)
                        .distinct()
                        .forEach(classifierID -> {
                            int[] numberOfErrorSamples = new int[] { 0 };
                            int[] numberOfSamples = new int[] { 0 };
                            data.stream()
                                    .filter(i -> i.label().equals(label) && i.functionId() == classifierID)
                                    .forEach(instance -> {
                                        if ((instance.value() >= 0.5 && !integratedPredictions.get(label).get(instance.id()))
                                                || (instance.value() < 0.5 && integratedPredictions.get(label).get(instance.id())))
                                            numberOfErrorSamples[0]++;
                                        numberOfSamples[0]++;
                                    });
                            errorRatesInstances.add(new ErrorRates.Instance(label, classifierID, numberOfErrorSamples[0] / (double) numberOfSamples[0]));
                        });
        integratedData = new Integrator.Data<>(integratedInstances);
        errorRates = new ErrorRates(errorRatesInstances);
        needsInference = false;
    }
}
