package org.platanios.experiment.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.experiment.classification.constraint.DataSets;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.reflection.ClassifierOutputsIntegrationResults;
import org.platanios.learn.classification.reflection.LogicErrorEstimation;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Emmanouil Antonios Platanios
 */
public class LogicErrorEstimationExperiment {
    private static final Logger logger = LogManager.getLogger("Classification / Reflection / Error Estimation Experiment");

    private final Set<Constraint> constaints;
    private final Map<Label, Set<Integer>> labelClassifiers;
    private final Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet;

    public LogicErrorEstimationExperiment(Set<Constraint> constaints,
                                          Map<Label, Set<Integer>> labelClassifiers,
                                          Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet) {
        this.constaints = constaints;
        this.labelClassifiers = labelClassifiers;
        this.dataSet = dataSet;
    }

    public void runExperiment() {
        LogicErrorEstimation logicErrorEstimation =
                new LogicErrorEstimation.Builder(labelClassifiers, dataSet)
                        .addConstraints(constaints)
                        .logProgress(true)
                        .build();
        ClassifierOutputsIntegrationResults results = logicErrorEstimation.estimateErrorRates();
        logger.info("Finished!");
    }

    public static void main(String[] args) {
        Set<Constraint> constraints = DataSets.importConstraints(args[0] + "/constraints.txt");
        NELLDataPreprocessing.Data data = NELLDataPreprocessing.aggregatePredictions(args[0]);
        Map<Label, Set<Integer>> labelClassifiers = new HashMap<>();
        BiMap<String, Integer> labelClassifierNamesMap = HashBiMap.create();
        AtomicInteger numberOfClassifiers = new AtomicInteger(0);
        data.getClassifierNames().forEach(classifierName -> labelClassifierNamesMap.put(classifierName, numberOfClassifiers.getAndIncrement()));
        data.getCategoryNames().forEach(category -> {
            Label label = new Label(category);
            Set<Integer> classifiers = new HashSet<>();
            for (int classifierIndex = 0; classifierIndex < numberOfClassifiers.get(); classifierIndex++)
                classifiers.add(classifierIndex);
            labelClassifiers.put(label, classifiers);
        });
        Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet = new HashMap<>();
        for (Map.Entry<String, Map<String, Map<String, Double>>> nounPhraseEntry : data.getPredictions().entrySet()) {
            Map<Label, Map<Integer, Double>> dataInstanceDataSet = new HashMap<>();
            for (Map.Entry<String, Map<String, Double>> categoryEntry : nounPhraseEntry.getValue().entrySet()) {
                Map<Integer, Double> categoryDataSet = new HashMap<>();
                for (Map.Entry<String, Double> classifierEntry : categoryEntry.getValue().entrySet())
                    categoryDataSet.put(labelClassifierNamesMap.get(classifierEntry.getKey()), classifierEntry.getValue());
                dataInstanceDataSet.put(new Label(categoryEntry.getKey()), categoryDataSet);
            }
            dataSet.put(new DataInstance<>(nounPhraseEntry.getKey(), null), dataInstanceDataSet);
        }
        LogicErrorEstimationExperiment experiment = new LogicErrorEstimationExperiment(constraints, labelClassifiers, dataSet);
        experiment.runExperiment();
    }
}
