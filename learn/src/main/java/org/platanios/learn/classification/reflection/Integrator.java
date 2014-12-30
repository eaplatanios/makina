package org.platanios.learn.classification.reflection;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.TrainableClassifier;
import org.platanios.learn.data.MultiViewPredictedDataInstance;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Integrator<T extends Vector, S> {
    private static final Logger logger = LogManager.getLogger("Classification / Integrator");

    private List<TrainableClassifier<T, S>> classifiers;
    private DataSelectionMethod dataSelectionMethod;
    private double dataSelectionParameter;
    private ExecutorService taskExecutor;
    private String workingDirectory;
    private boolean saveModelsOnEveryIteration;
    private boolean useDifferentFilePerIteration;

    private List<MultiViewPredictedDataInstance<T, S>> labeledDataInstances;
    private List<MultiViewPredictedDataInstance<T, S>> unlabeledDataInstances;
    private int iterationNumber = 1;

    public static class Builder<T extends Vector, S> {
        private List<TrainableClassifier<T, S>> classifiers;
        private List<MultiViewPredictedDataInstance<T, S>> dataInstances;

        private DataSelectionMethod dataSelectionMethod = DataSelectionMethod.FIXED_PROPORTION;
        private double dataSelectionParameter = 0.1;
        private int numberOfThreads = Runtime.getRuntime().availableProcessors();
        private String workingDirectory = "Integrator Directory";
        private boolean saveModelsOnEveryIteration = true;
        private boolean useDifferentFilePerIteration = true;

        public Builder() {
            classifiers = new ArrayList<>();
            dataInstances = new ArrayList<>();
        }

        @SuppressWarnings("unchecked")
        public Builder(String modelsFileAbsolutePath, boolean resumeTraining) {
            classifiers = new ArrayList<>();
            dataInstances = new ArrayList<>();
            if (resumeTraining) {
                File inputFile = new File(modelsFileAbsolutePath);
                workingDirectory = inputFile.getParentFile().getAbsolutePath();
                try {
                    ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(inputFile));
                    int numberOfClassifiers = objectInputStream.readInt();
                    for (int i = 0; i < numberOfClassifiers; i++)
                        classifiers.add((TrainableClassifier<T, S>) objectInputStream.readObject());
                    objectInputStream.close();
                } catch (IOException|ClassNotFoundException e) {
                    logger.error("Could load the classifier models from the file \""
                                         + inputFile.getAbsolutePath() + "\"!");
                }
            }
        }

        public Builder addClassifier(TrainableClassifier<T, S> classifier) {
            classifiers.add(classifier);
            return this;
        }

        public Builder addDataInstance(MultiViewPredictedDataInstance<T, S> dataInstance) {
            dataInstances.add(dataInstance);
            return this;
        }

        public Builder addDataInstances(List<MultiViewPredictedDataInstance<T, S>> dataInstances) {
            this.dataInstances.addAll(dataInstances);
            return this;
        }

        public Builder dataSelectionMethod(DataSelectionMethod dataSelectionMethod) {
            this.dataSelectionMethod = dataSelectionMethod;
            return this;
        }

        public Builder dataSelectionParameter(double dataSelectionParameter) {
            this.dataSelectionParameter = dataSelectionParameter;
            return this;
        }

        public Builder numberOfThreads(int numberOfThreads) {
            this.numberOfThreads = numberOfThreads;
            return this;
        }

        public Builder workingDirectory(String workingDirectory) {
            this.workingDirectory = workingDirectory;
            return this;
        }

        public Builder saveModelsOnEveryIteration(boolean saveModelsOnEveryIteration) {
            this.saveModelsOnEveryIteration = saveModelsOnEveryIteration;
            return this;
        }

        public Builder useDifferentFilePerIteration(boolean useDifferentFilePerIteration) {
            this.useDifferentFilePerIteration = useDifferentFilePerIteration;
            return this;
        }

        public Integrator<T, S> build() {
            return new Integrator<T, S>(this);
        }
    }

    private Integrator(Builder<T, S> builder) {
        classifiers = builder.classifiers;
        dataSelectionMethod = builder.dataSelectionMethod;
        dataSelectionParameter = builder.dataSelectionParameter;
        taskExecutor = Executors.newFixedThreadPool(builder.numberOfThreads);
        workingDirectory = builder.workingDirectory;
        saveModelsOnEveryIteration = builder.saveModelsOnEveryIteration;
        useDifferentFilePerIteration = builder.useDifferentFilePerIteration;
        initializeWorkingDirectory();
        labeledDataInstances = new ArrayList<>();
        unlabeledDataInstances = new ArrayList<>();
        for (MultiViewPredictedDataInstance<T, S> dataInstance : builder.dataInstances) {
            if (dataInstance.label() != null) {
                labeledDataInstances.add(dataInstance);
            } else {
                unlabeledDataInstances.add(dataInstance);
            }
        }
    }

    private void initializeWorkingDirectory() {
        File directory = new File(workingDirectory);
        if (!directory.exists() && !directory.mkdirs()) {
            logger.error("Unable to create directory " + directory.getAbsolutePath());
        }
    }

    public void trainClassifiers() {
        List<Callable<Boolean>> classifierTrainingTasks = new ArrayList<>();
        for (int i = 0; i < classifiers.size(); i++) {
            TrainableClassifier<T, S> classifier = classifiers.get(i);
//            List<PredictedDataInstance<T, S>> trainingData =
//                    DataInstances.getSingleViewDataInstances(labeledDataInstances, i);
//            classifierTrainingTasks.add(() -> classifier.train(trainingData));
        }
        try {
            taskExecutor.invokeAll(classifierTrainingTasks);
        } catch (InterruptedException e) {
            logger.error("Execution was interrupted while training the classifiers.");
        }
    }

    public void makePredictions() {
        List<Callable<List<PredictedDataInstance<T, S>>>> classifierPredictionTasks = new ArrayList<>();
        for (int i = 0; i < classifiers.size(); i++) {
            TrainableClassifier<T, S> classifier = classifiers.get(i);
//            List<PredictedDataInstance<T, S>> testingData =
//                    DataInstances.getSingleViewDataInstances(unlabeledDataInstances, i);
//            classifierPredictionTasks.add(() -> classifier.predict(testingData));
        }
        try {
            List<Future<List<PredictedDataInstance<T, S>>>> predictionResults =
                    taskExecutor.invokeAll(classifierPredictionTasks);
//            for (int i = 0; i < classifiers.size(); i++) {
//                List<PredictedDataInstance<T, S>> dataInstances = predictionResults.get(i).get();
//                for (int j = 0; j < dataInstances.size(); j++) {
//                    if (dataInstances.get(j).probability() > unlabeledDataInstances.get(j).probability()) {
//                        unlabeledDataInstances.set(i, new MultiViewPredictedDataInstance<>(
//                                null,
//                                unlabeledDataInstances.get(j),
//                                dataInstances.get(j).label(),
//                                null,
//                                dataInstances.get(j).probability())
//                        );
//                    }
//                }
//            }
        } catch (InterruptedException e) {
            logger.error("Execution was interrupted while making predictions with the classifiers.");
        }
//        } catch (ExecutionException e) {
//            logger.error("Something went wrong while making predictions with the classifiers.");
//        }
    }

    public void transferData() {
        dataSelectionMethod.transferData(labeledDataInstances, unlabeledDataInstances, dataSelectionParameter);
    }

    public void performSingleIteration() {
        trainClassifiers();
        makePredictions();
        transferData();
        if (saveModelsOnEveryIteration)
            saveModels(useDifferentFilePerIteration);
        iterationNumber++;
    }

    private void saveModels(boolean useDifferentFilePerIteration) {
        File outputFile;
        if (useDifferentFilePerIteration)
            outputFile = new File(workingDirectory + File.separator + "Iteration_" + iterationNumber + ".integrator");
        else
            outputFile = new File(workingDirectory + File.separator + "Models.integrator");

        try {
            if (!outputFile.exists() && outputFile.createNewFile())
                logger.error("Could not create the file \"" + outputFile.getAbsolutePath() + "\" to store the models!");
            ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(outputFile, false));
            outputStream.writeInt(iterationNumber);
            outputStream.writeInt(classifiers.size());
            for (TrainableClassifier<T, S> classifier : classifiers)
                outputStream.writeObject(classifier);
            outputStream.close();
        } catch (IOException e) {
            logger.error("Could not create or open the file \""
                                 + outputFile.getAbsolutePath() + "\" to store the models!");
        }
    }

    @SuppressWarnings("unchecked")
    private void loadModels(String modelsFileAbsolutePath) {
        classifiers = new ArrayList<>();
        File inputFile = new File(modelsFileAbsolutePath);
        workingDirectory = inputFile.getParentFile().getAbsolutePath();
        try {
            ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(inputFile));
            iterationNumber = inputStream.readInt();
            int numberOfClassifiers = inputStream.readInt();
            for (int i = 0; i < numberOfClassifiers; i++)
                classifiers.add((TrainableClassifier<T, S>) inputStream.readObject());
            inputStream.close();
        } catch (IOException|ClassNotFoundException e) {
            logger.error("Could load the classifier models from the file \"" + inputFile.getAbsolutePath() + "\"!");
        }
    }

    public enum DataSelectionMethod {
        FIXED_PROPORTION {
            @Override
            public <T extends Vector, S> void transferData(
                    List<MultiViewPredictedDataInstance<T, S>> labeledDataInstances,
                    List<MultiViewPredictedDataInstance<T, S>> unlabeledDataInstances,
                    double proportionToTransfer
            ) {
                unlabeledDataInstances
                        .parallelStream()
                        .sorted((i1, i2) -> -Double.compare(i1.probability(), i2.probability()));
                int numberOfPredictionsToTransfer =
                        (int) Math.floor(proportionToTransfer * unlabeledDataInstances.size());
                for (int i = 0; i < numberOfPredictionsToTransfer; i++) {
                    labeledDataInstances.add(unlabeledDataInstances.get(0));
                    unlabeledDataInstances.remove(0);
                }
            }
        },
        PROBABILITY_THRESHOLD {
            @Override
            public <T extends Vector, S> void transferData(
                    List<MultiViewPredictedDataInstance<T, S>> labeledDataInstances,
                    List<MultiViewPredictedDataInstance<T, S>> unlabeledDataInstances,
                    double probabilityThreshold
            ) {
                unlabeledDataInstances
                        .parallelStream()
                        .sorted((i1, i2) -> -Double.compare(i1.probability(), i2.probability()));
                for (int i = 0; i < unlabeledDataInstances.size(); i++) {
                    if (unlabeledDataInstances.get(0).probability() >= probabilityThreshold) {
                        labeledDataInstances.add(unlabeledDataInstances.get(0));
                        unlabeledDataInstances.remove(0);
                    } else {
                        break;
                    }
                }
            }
        };

        public abstract <T extends Vector, S> void transferData(
                List<MultiViewPredictedDataInstance<T, S>> labeledDataInstances,
                List<MultiViewPredictedDataInstance<T, S>> unlabeledDataInstances,
                double parameter
        );
    }
}
