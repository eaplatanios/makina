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
import org.platanios.learn.math.matrix.Vector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class TuffyLogicIntegrator {
    private static final Logger logger = LogManager.getLogger("Classification / Tuffy Logic Integrator");

    private final Set<Label> labels;
    private final Map<Label, Set<Integer>> labelClassifiers;
    private final Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSet;
    private final Set<Constraint> constraints;
    private final boolean estimateErrorRates;
    private final boolean logProgress;
    private final String workingDirectory;
    private final BiMap<Long, DataInstance<Vector>> instanceKeysMap;
    private final BiMap<Long, Label> labelKeysMap;
    private final BiMap<Long, Integer> classifierKeysMap;

    public static class Builder {
        private final Set<Label> labels;
        private final Map<Label, Set<Integer>> labelClassifiers;
        private final Map<DataInstance<Vector>, Map<Label, Boolean>> fixedDataSet;
        private final Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> predictionsDataSet;

        private Set<Constraint> constraints = new HashSet<>();
        private boolean estimateErrorRates = false;
        private boolean logProgress = false;
        private String workingDirectory = "/temp";

        public Builder(Map<Label, Set<Integer>> labelClassifiers,
                       Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> predictionsDataSet) {
            this(labelClassifiers, null, predictionsDataSet);
        }

        public Builder(Map<Label, Set<Integer>> labelClassifiers,
                       Map<DataInstance<Vector>, Map<Label, Boolean>> fixedDataSet,
                       Map<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> predictionsDataSet) {
            this.labels = labelClassifiers.keySet();
            this.labelClassifiers = labelClassifiers;
            this.fixedDataSet = fixedDataSet;
            this.predictionsDataSet = predictionsDataSet;
        }

        public Builder addConstraint(MutualExclusionConstraint constraint) {
            constraints.add(constraint);
            return this;
        }

        public Builder addConstraint(SubsumptionConstraint constraint) {
            constraints.add(constraint);
            return this;
        }

        public Builder addConstraints(Set<Constraint> constraints) {
            this.constraints.addAll(constraints);
            return this;
        }

        public Builder estimateErrorRates(boolean estimateErrorRates) {
            this.estimateErrorRates = estimateErrorRates;
            return this;
        }

        public Builder logProgress(boolean logProgress) {
            this.logProgress = logProgress;
            return this;
        }

        public Builder workingDirectory(String workingDirectory) {
            this.workingDirectory = workingDirectory;
            return this;
        }

        public TuffyLogicIntegrator build() {
            return new TuffyLogicIntegrator(this);
        }
    }

    private TuffyLogicIntegrator(Builder builder) {
        labels = builder.labels;
        labelClassifiers = builder.labelClassifiers;
        dataSet = builder.predictionsDataSet;
        constraints = builder.constraints;
        estimateErrorRates = builder.estimateErrorRates;
        logProgress = builder.logProgress;
        workingDirectory = builder.workingDirectory;
        instanceKeysMap = HashBiMap.create(dataSet.size());
        labelKeysMap = HashBiMap.create(labels.size());
        classifierKeysMap = HashBiMap.create(
                (int) labelClassifiers.values().stream().flatMap(Collection::stream).count()
        );
        final long[] currentInstanceKey = { 0 };
        final long[] currentLabelKey = { 0 };
        final long[] currentClassifierKey = { 0 };
        if (builder.fixedDataSet != null)
            builder.fixedDataSet.keySet().forEach(instance -> {
                if (!instanceKeysMap.containsValue(instance))
                    instanceKeysMap.put(currentInstanceKey[0]++, instance);
            });
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
        try {
            Files.copy(TuffyLogicIntegrator.class.getResourceAsStream("./logic_integrator.mln"),
                       Paths.get(workingDirectory + "/logic_integrator.mln"));
            Files.copy(TuffyLogicIntegrator.class.getResourceAsStream("./tuffy.jar"),
                       Paths.get(workingDirectory + "/tuffy.jar"));
            Files.copy(TuffyLogicIntegrator.class.getResourceAsStream("./tuffy.conf"),
                       Paths.get(workingDirectory + "/tuffy.conf"));
            File evidenceFile = new File(workingDirectory + "/evidence.db");
            evidenceFile.getParentFile().mkdirs();
            FileWriter evidenceFileWriter = new FileWriter(evidenceFile);
            labels.forEach(label -> {
                try {
                    evidenceFileWriter.write("EqualLabels(" + labelKeysMap.inverse().get(label) + ", " + labelKeysMap.inverse().get(label) + ")\n");
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
            for (Constraint constraint : constraints) {
                if (constraint instanceof MutualExclusionConstraint) {
                    List<Label> labelsList = new ArrayList<>(((MutualExclusionConstraint) constraint).getLabels());
                    for (int label1Index = 0; label1Index < labelsList.size(); label1Index++)
                        for (int label2Index = label1Index + 1; label2Index < labelsList.size(); label2Index++)
                            evidenceFileWriter.write("MutuallyExclusiveLabels(" + labelKeysMap.inverse().get(labelsList.get(label1Index)) + ", " + labelKeysMap.inverse().get(labelsList.get(label2Index)) + ")\n");
                } else if (constraint instanceof SubsumptionConstraint) {
                    Label parentLabel = ((SubsumptionConstraint) constraint).getParentLabel();
                    Label childLabel = ((SubsumptionConstraint) constraint).getChildLabel();
                    evidenceFileWriter.write("SubsumingLabels(" + labelKeysMap.inverse().get(parentLabel) + ", " + labelKeysMap.inverse().get(childLabel) + ")\n");
                }
            }
            if (builder.fixedDataSet != null)
                for (Map.Entry<DataInstance<Vector>, Map<Label, Boolean>> dataSetEntry : builder.fixedDataSet.entrySet())
                    for (Map.Entry<Label, Boolean> dataInstanceEntry : dataSetEntry.getValue().entrySet())
                        evidenceFileWriter.write("Label(" + instanceKeysMap.inverse().get(dataSetEntry.getKey()) + ", "
                                                         + labelKeysMap.inverse().get(dataInstanceEntry.getKey()) + ")\n");
            for (Map.Entry<DataInstance<Vector>, Map<Label, Map<Integer, Double>>> dataSetEntry : dataSet.entrySet())
                for (Map.Entry<Label, Map<Integer, Double>> dataInstanceEntry : dataSetEntry.getValue().entrySet())
                    for (Map.Entry<Integer, Double> classifierEntry : dataInstanceEntry.getValue().entrySet())
                        evidenceFileWriter.write("LabelPrediction(" + instanceKeysMap.inverse().get(dataSetEntry.getKey()) + ", "
                                                         + classifierKeysMap.inverse().get(classifierEntry.getKey()) + ", "
                                                         + labelKeysMap.inverse().get(dataInstanceEntry.getKey()) + ")\n");
            evidenceFileWriter.close();
            File queryFile = new File(workingDirectory + "/query.db");
            queryFile.getParentFile().mkdirs();
            FileWriter queryFileWriter = new FileWriter(queryFile);
            for (long classifierKey : classifierKeysMap.keySet())
                for (long labelKey : labelKeysMap.keySet())
                    queryFileWriter.write("ErrorRate(" + classifierKey + ", " + labelKey + ")\n");
//            for (long instanceKey : instanceKeysMap.keySet())
//                for (long labelKey : labelKeysMap.keySet())
//                    queryFileWriter.write("Label(" + instanceKey + ", " + labelKey + ")\n");
            queryFileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ClassifierOutputsIntegrationResults integratePredictions() {
        if (logProgress)
            logger.info("Starting inference...");
        Map<DataInstance<Vector>, Map<Label, Double>> integratedDataSet = new HashMap<>();
        Map<Label, Map<Integer, Double>> errorRates = new HashMap<>();
        for (DataInstance<Vector> instance : dataSet.keySet())
            integratedDataSet.put(instance, new HashMap<>());
        for (Label label : labels)
            errorRates.put(label, new HashMap<>());
        try {
            logger.info("Started running Tuffy.");
            ProcessBuilder pb = new ProcessBuilder("java",
                                                   "-Xmx12G",
                                                   "-jar", workingDirectory + "/tuffy.jar",
                                                   "-conf", workingDirectory + "/tuffy.conf",
                                                   "-i", workingDirectory + "/logic_integrator.mln",
                                                   "-e", workingDirectory + "/evidence.db",
                                                   "-queryFile", workingDirectory + "/query.db",
                                                   "-r", workingDirectory + "/results.db",
                                                   "-marginal");
            pb.redirectOutput(ProcessBuilder.Redirect.INHERIT);
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);
            Process p = pb.start();
            p.waitFor();
            logger.info("Finished running Tuffy.");
            Files.newBufferedReader(Paths.get(workingDirectory + "/results.db")).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                String[] argumentParts = lineParts[1].split("\"");
                if (lineParts[1].startsWith("ErrorRate")) {
                    int classifierId = classifierKeysMap.get(Long.parseLong(argumentParts[1]));
                    Label label = labelKeysMap.get(Long.parseLong(argumentParts[3]));
                    errorRates.get(label).put(classifierId, Double.parseDouble(lineParts[0]));
                } else if (lineParts[1].startsWith("Label")) {
                    DataInstance<Vector> instance = instanceKeysMap.get(Long.parseLong(argumentParts[1]));
                    Label label = labelKeysMap.get(Long.parseLong(argumentParts[3]));
                    integratedDataSet.get(instance).put(label, Double.parseDouble(lineParts[0]));
                }
            });
        } catch (IOException|InterruptedException e) {
            e.printStackTrace();
        }
        return new ClassifierOutputsIntegrationResults(integratedDataSet, errorRates);
    }
}