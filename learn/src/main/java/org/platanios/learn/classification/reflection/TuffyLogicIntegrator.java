package org.platanios.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.Label;
import org.platanios.learn.classification.constraint.Constraint;
import org.platanios.learn.classification.constraint.MutualExclusionConstraint;
import org.platanios.learn.classification.constraint.SubsumptionConstraint;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class TuffyLogicIntegrator extends Integrator {
    private static final Logger logger = LogManager.getLogger("Classification / Tuffy Logic Integrator");

    private final Set<Label> labels;
    private final Set<Integer> classifiers;
    private final Set<Constraint> constraints;
    private final boolean logProgress;
    private final String workingDirectory;
    private final BiMap<Long, Integer> instanceKeysMap;
    private final BiMap<Long, Label> labelKeysMap;
    private final BiMap<Long, Integer> classifierKeysMap;

    private boolean needsInference = true;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends Integrator.AbstractBuilder<T> {
        private final Set<Label> labels = new HashSet<>();
        private final Set<Integer> classifiers = new HashSet<>();

        private final Integrator.Data<Integrator.Data.ObservedInstance> observedData;

        private Set<Constraint> constraints = new HashSet<>();
        private boolean logProgress = false;
        private String workingDirectory = "/temp";

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

        public T logProgress(boolean logProgress) {
            this.logProgress = logProgress;
            return self();
        }

        public T workingDirectory(String workingDirectory) {
            this.workingDirectory = workingDirectory;
            return self();
        }

        public TuffyLogicIntegrator build() {
            return new TuffyLogicIntegrator(this);
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

    private TuffyLogicIntegrator(AbstractBuilder<?> builder) {
        super(builder);
        labels = builder.labels;
        classifiers = builder.classifiers;
        constraints = builder.constraints;
        logProgress = builder.logProgress;
        workingDirectory = builder.workingDirectory;
        instanceKeysMap = HashBiMap.create(data.size());
        labelKeysMap = HashBiMap.create(labels.size());
        classifierKeysMap = HashBiMap.create((int) classifiers.stream().count());
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
        try {
            Files.copy(TuffyLogicIntegrator.class.getResourceAsStream("./logic_integrator.mln"),
                       Paths.get(workingDirectory + "/logic_integrator.mln"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            Files.copy(TuffyLogicIntegrator.class.getResourceAsStream("./tuffy.jar"),
                       Paths.get(workingDirectory + "/tuffy.jar"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            Files.copy(TuffyLogicIntegrator.class.getResourceAsStream("./tuffy.conf"),
                       Paths.get(workingDirectory + "/tuffy.conf"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
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
            if (builder.observedData != null)
                for (Integrator.Data.ObservedInstance instance : builder.observedData)
                    evidenceFileWriter.write("Label(" + instanceKeysMap.inverse().get(instance.id()) + ", "
                                                     + labelKeysMap.inverse().get(instance.label()) + ")\n");
            for (Integrator.Data.PredictedInstance instance : data)
                evidenceFileWriter.write(instance.value() + "\tLabelPrediction("
                                                 + instanceKeysMap.inverse().get(instance.id()) + ", "
                                                 + classifierKeysMap.inverse().get(instance.functionId()) + ", "
                                                 + labelKeysMap.inverse().get(instance.label()) + ")\n");
            evidenceFileWriter.close();
            File queryFile = new File(workingDirectory + "/query.db");
            queryFile.getParentFile().mkdirs();
            FileWriter queryFileWriter = new FileWriter(queryFile);
            for (long classifierKey : classifierKeysMap.keySet())
                for (long labelKey : labelKeysMap.keySet())
                    queryFileWriter.write("ErrorRate(" + classifierKey + ", " + labelKey + ")\n");
            for (long instanceKey : instanceKeysMap.keySet())
                for (long labelKey : labelKeysMap.keySet())
                    queryFileWriter.write("Label(" + instanceKey + ", " + labelKey + ")\n");
            queryFileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
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
        List<Integrator.Data.PredictedInstance> integratedInstances = new ArrayList<>();
        List<ErrorRates.Instance> errorRatesInstances = new ArrayList<>();
        try {
            logger.info("Started running Tuffy.");
            ProcessBuilder pb = new ProcessBuilder("java",
                                                   "-Xmx12G",
                                                   "-jar", workingDirectory + "/tuffy.jar",
                                                   "-conf", workingDirectory + "/tuffy.conf",
//                                                   "-learnwt",
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
                if (lineParts[1].startsWith("Label")) {
                    String[] argumentParts = lineParts[1].substring(6, lineParts[1].length() - 1).split(", ");
                    Integer instanceID = instanceKeysMap.get(Long.parseLong(argumentParts[0]));
                    Label label = labelKeysMap.get(Long.parseLong(argumentParts[1]));
                    integratedInstances.add(new Integrator.Data.PredictedInstance(
                            instanceID, label, -1, Double.parseDouble(lineParts[0]))
                    );
                } else if (lineParts[1].startsWith("ErrorRate")) {
                    String[] argumentParts = lineParts[1].substring(10, lineParts[1].length() - 1).split(", ");
                    int classifierId = classifierKeysMap.get(Long.parseLong(argumentParts[0]));
                    Label label = labelKeysMap.get(Long.parseLong(argumentParts[1]));
                    errorRatesInstances.add(new ErrorRates.Instance(
                            label, classifierId, Double.parseDouble(lineParts[0]))
                    );
                }
            });
        } catch (IOException|InterruptedException e) {
            e.printStackTrace();
        }
        integratedData = new Integrator.Data<>(integratedInstances);
        errorRates = new ErrorRates(errorRatesInstances);
        needsInference = false;
    }
}
