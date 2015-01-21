package org.platanios.learn.classification.reflection;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.classification.Classifiers;
import org.platanios.learn.classification.TrainableClassifier;
import org.platanios.learn.data.*;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.serialization.UnsafeSerializationUtilities;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Integrator<T extends Vector, S> {
    private static final Logger logger = LogManager.getLogger("Classification / Integrator");

    private List<TrainableClassifier<T, S>> classifiers;
    private MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet;
    private MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet;
    private DataSelectionMethod dataSelectionMethod;
    private double dataSelectionParameter;
    private ExecutorService taskExecutor;
    private String workingDirectory;
    private boolean saveModelsOnEveryIteration;
    private boolean useDifferentFilePerIteration;

    private int iterationNumber = 1;

    public static class Builder<T extends Vector, S> {
        private List<TrainableClassifier<T, S>> classifiers;
        private MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet;
        private MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet;

        private int iterationNumber = 1;
        private DataSelectionMethod dataSelectionMethod = DataSelectionMethod.FIXED_PROPORTION;
        private double dataSelectionParameter = 0.1;
        private int numberOfThreads = Runtime.getRuntime().availableProcessors();
        private String workingDirectory = "Integrator Directory";
        private boolean saveModelsOnEveryIteration = true;
        private boolean useDifferentFilePerIteration = true;

        public Builder() {
            classifiers = new ArrayList<>();
        }

        @SuppressWarnings("unchecked")
        public Builder(String modelsFileAbsolutePath, boolean resumeTraining) {
            classifiers = new ArrayList<>();
            if (resumeTraining) {
                File inputFile = new File(modelsFileAbsolutePath);
                workingDirectory = inputFile.getParentFile().getAbsolutePath();
                try {
                    InputStream inputStream = new FileInputStream(inputFile);
                    iterationNumber = UnsafeSerializationUtilities.readInt(inputStream);
                    int numberOfClassifiers = UnsafeSerializationUtilities.readInt(inputStream);
                    for (int i = 0; i < numberOfClassifiers; i++)
                        classifiers.add((TrainableClassifier<T, S>) Classifiers.read(inputStream));
                    inputStream.close();
                } catch (IOException e) {
                    logger.error("Could load the classifier models from the file \""
                                         + inputFile.getAbsolutePath() + "\"!");
                }
            }
        }

        public Builder addClassifier(TrainableClassifier<T, S> classifier) {
            classifiers.add(classifier);
            return this;
        }

        public Builder labeledDataSet(MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet) {
            this.labeledDataSet = labeledDataSet;
            return this;
        }

        public Builder unlabeledDataSet(MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet) {
            this.unlabeledDataSet = unlabeledDataSet;
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
            return new Integrator<>(this);
        }
    }

    private Integrator(Builder<T, S> builder) {
        classifiers = builder.classifiers;
        labeledDataSet = builder.labeledDataSet;
        unlabeledDataSet = builder.unlabeledDataSet;
        dataSelectionMethod = builder.dataSelectionMethod;
        dataSelectionParameter = builder.dataSelectionParameter;
        taskExecutor = Executors.newFixedThreadPool(builder.numberOfThreads);
        workingDirectory = builder.workingDirectory;
        saveModelsOnEveryIteration = builder.saveModelsOnEveryIteration;
        useDifferentFilePerIteration = builder.useDifferentFilePerIteration;
        initializeWorkingDirectory();
        iterationNumber = builder.iterationNumber;
    }

    private void initializeWorkingDirectory() {
        File directory = new File(workingDirectory);
        if (!directory.exists() && !directory.mkdirs())
            logger.error("Unable to create directory " + directory.getAbsolutePath());
    }

    public void trainClassifiers() {
        List<Callable<Boolean>> classifierTrainingTasks = new ArrayList<>();
        for (int i = 0; i < classifiers.size(); i++) {
            TrainableClassifier<T, S> classifier = classifiers.get(i);
            DataSet<PredictedDataInstance<T, S>> trainingData =
                    (DataSet<PredictedDataInstance<T, S>>) labeledDataSet.getSingleViewDataSet(i);
            classifierTrainingTasks.add(() -> classifier.train(trainingData));
        }
        try {
            taskExecutor.invokeAll(classifierTrainingTasks);
        } catch (InterruptedException e) {
            logger.error("Execution was interrupted while training the classifiers.");
        }
    }

    public void makePredictions() {
        List<Callable<DataSet<PredictedDataInstance<T, S>>>> classifierPredictionTasks = new ArrayList<>();
        for (int i = 0; i < classifiers.size(); i++) {
            TrainableClassifier<T, S> classifier = classifiers.get(i);
            DataSet<PredictedDataInstance<T, S>> testingData =
                    (DataSet<PredictedDataInstance<T, S>>) unlabeledDataSet.getSingleViewDataSet(i);
            classifierPredictionTasks.add(() -> classifier.predict(testingData));
        }
        try {
            List<Future<DataSet<PredictedDataInstance<T, S>>>> predictionResults =
                    taskExecutor.invokeAll(classifierPredictionTasks);
            for (int i = 0; i < classifiers.size(); i++) {
                DataSet<PredictedDataInstance<T, S>> dataSet = predictionResults.get(i).get();
                for (int j = 0; j < dataSet.size(); j++) {
                    if (dataSet.get(j).probability() > unlabeledDataSet.get(j).probability()) { // Keep the highest probability prediction / Most confident prediction
                        unlabeledDataSet.set(i, new MultiViewPredictedDataInstance<>(
                                null,
                                unlabeledDataSet.get(j).features(),
                                dataSet.get(j).label(),
                                null,
                                dataSet.get(j).probability())
                        );
                    }
                }
            }
        } catch (InterruptedException e) {
            logger.error("Execution was interrupted while making predictions with the classifiers.");
        } catch (ExecutionException e) {
            logger.error("Something went wrong while making predictions with the classifiers.");
        }
    }

    public void transferData() {
        dataSelectionMethod.transferData(labeledDataSet, unlabeledDataSet, dataSelectionParameter);
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
            outputFile = new File(workingDirectory + File.separator
                                          + "Iteration_" + iterationNumber + "_Models.integrator");
        else
            outputFile = new File(workingDirectory + File.separator + "Models.integrator");

        try {
            if (!outputFile.exists() && outputFile.createNewFile())
                logger.error("Could not create the file \"" + outputFile.getAbsolutePath() + "\" to store the models!");
            OutputStream outputStream = new FileOutputStream(outputFile, false);
            UnsafeSerializationUtilities.writeInt(outputStream, iterationNumber);
            UnsafeSerializationUtilities.writeInt(outputStream, classifiers.size());
            for (TrainableClassifier<T, S> classifier : classifiers)
                classifier.write(outputStream, true);
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
            InputStream inputStream = new FileInputStream(inputFile);
            iterationNumber = UnsafeSerializationUtilities.readInt(inputStream);
            int numberOfClassifiers = UnsafeSerializationUtilities.readInt(inputStream);
            for (int i = 0; i < numberOfClassifiers; i++)
                classifiers.add((TrainableClassifier<T, S>) Classifiers.read(inputStream));
            inputStream.close();
        } catch (IOException e) {
            logger.error("Could load the classifier models from the file \"" + inputFile.getAbsolutePath() + "\"!");
        }
    }

    public enum DataSelectionMethod {
        FIXED_PROPORTION {
            @Override
            public <T extends Vector, S> void transferData(
                    MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet,
                    MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet,
                    double proportionToTransfer
            ) {
                unlabeledDataSet.sort((i1, i2) -> -Double.compare(i1.probability(), i2.probability()));
                int numberOfPredictionsToTransfer = (int) Math.floor(proportionToTransfer * unlabeledDataSet.size());
                for (int i = 0; i < numberOfPredictionsToTransfer; i++) {
                    labeledDataSet.add(unlabeledDataSet.get(0));
                    unlabeledDataSet.remove(0);
                }
            }
        },
        PROBABILITY_THRESHOLD {
            @Override
            public <T extends Vector, S> void transferData(
                    MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet,
                    MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet,
                    double probabilityThreshold
            ) {
                unlabeledDataSet.sort((i1, i2) -> -Double.compare(i1.probability(), i2.probability()));
                for (int i = 0; i < unlabeledDataSet.size(); i++) {
                    if (unlabeledDataSet.get(0).probability() >= probabilityThreshold) {
                        labeledDataSet.add(unlabeledDataSet.get(0));
                        unlabeledDataSet.remove(0);
                    } else {
                        break;
                    }
                }
            }
        };

        public abstract <T extends Vector, S> void transferData(
                MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet,
                MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet,
                double parameter
        );
    }
}
