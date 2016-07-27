package makina.learn.classification.reflection;

import com.lmax.disruptor.EventHandler;
import com.lmax.disruptor.RingBuffer;
import com.lmax.disruptor.dsl.Disruptor;
import makina.learn.classification.Classifiers;
import makina.learn.classification.Label;
import makina.learn.classification.TrainableClassifier;
import makina.learn.data.DataSet;
import makina.learn.data.MultiViewDataSet;
import makina.learn.data.MultiViewPredictedDataInstance;
import makina.learn.data.PredictedDataInstance;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.math.matrix.Vector;
import makina.utilities.UnsafeSerializationUtilities;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CoTrainingIntegrator<T extends Vector, S> {
    private static final Logger logger = LogManager.getLogger("Classification / Co-Training Integrator");

    private List<TrainableClassifier<T, S>> classifiers;
    private MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet;
    private MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet;
    private RingBuffer<CompletedIterationEvent> ringBuffer;
    private CoTrainingMethod coTrainingMethod;
    private DataSelectionMethod dataSelectionMethod;
    private double dataSelectionParameter;
    private ExecutorService taskExecutor;
    private String workingDirectory;
    private boolean saveModelsOnEveryIteration;
    private boolean useDifferentFilePerIteration;
    private double[] errorRates;

    private int iterationNumber = 1;

    public static class Builder<T extends Vector, S> {
        private List<TrainableClassifier<T, S>> classifiers;
        private String workingDirectory;
        private MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet;
        private MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet;
        private EventHandler<CoTrainingIntegrator.CompletedIterationEvent>[] completedIterationEventHandlers;

        private int iterationNumber = 1;
        private CoTrainingMethod coTrainingMethod = CoTrainingMethod.CO_TRAINING;
        private DataSelectionMethod dataSelectionMethod = DataSelectionMethod.FIXED_PROPORTION;
        private double dataSelectionParameter = 0.1;
        private int numberOfThreads = Runtime.getRuntime().availableProcessors();
        private boolean saveModelsOnEveryIteration = true;
        private boolean useDifferentFilePerIteration = true;

        public Builder(String workingDirectory) {
            classifiers = new ArrayList<>();
            this.workingDirectory = workingDirectory;
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

        public Builder completedIterationEventHandlers(EventHandler<CoTrainingIntegrator.CompletedIterationEvent>... eventHandlers) {
            completedIterationEventHandlers = eventHandlers;
            return this;
        }

        public Builder coTrainingMethod(CoTrainingMethod coTrainingMethod) {
            this.coTrainingMethod = coTrainingMethod;
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

        public Builder saveModelsOnEveryIteration(boolean saveModelsOnEveryIteration) {
            this.saveModelsOnEveryIteration = saveModelsOnEveryIteration;
            return this;
        }

        public Builder useDifferentFilePerIteration(boolean useDifferentFilePerIteration) {
            this.useDifferentFilePerIteration = useDifferentFilePerIteration;
            return this;
        }

        public CoTrainingIntegrator<T, S> build() {
            return new CoTrainingIntegrator<>(this);
        }
    }

    private Map<String, S> trueLabels = new HashMap<>();

    @SuppressWarnings("unchecked")
    private CoTrainingIntegrator(Builder<T, S> builder) {
        classifiers = builder.classifiers;
        labeledDataSet = builder.labeledDataSet;
        unlabeledDataSet = builder.unlabeledDataSet;
        for (MultiViewPredictedDataInstance<T, S> dataInstance : unlabeledDataSet)
            trueLabels.put(dataInstance.name(), dataInstance.label());
        coTrainingMethod = builder.coTrainingMethod;
        dataSelectionMethod = builder.dataSelectionMethod;
        dataSelectionParameter = builder.dataSelectionParameter;
        taskExecutor = Executors.newFixedThreadPool(builder.numberOfThreads);
        workingDirectory = builder.workingDirectory;
        saveModelsOnEveryIteration = builder.saveModelsOnEveryIteration;
        useDifferentFilePerIteration = builder.useDifferentFilePerIteration;
        errorRates = new double[classifiers.size()];
        initializeWorkingDirectory();
        iterationNumber = builder.iterationNumber;
        Disruptor<CompletedIterationEvent> disruptor =
                new Disruptor<>(CompletedIterationEvent::new, 16, Executors.newCachedThreadPool());
        disruptor.handleEventsWith(builder.completedIterationEventHandlers);
        disruptor.start();
        ringBuffer = disruptor.getRingBuffer();
    }

    public int getIterationNumber() {
        return iterationNumber;
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
            coTrainingMethod.updatePredictions(predictionResults, this);
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
        ringBuffer.publishEvent((event, sequence, classifiers) -> {
            event.setIterationNumber(iterationNumber);
            event.setClassifiers(classifiers);
            event.setErrorRates(errorRates);
        }, classifiers);
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
            if (!outputFile.exists() && !outputFile.createNewFile())
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

    public class CompletedIterationEvent {
        private int iterationNumber;
        private List<TrainableClassifier<T, S>> classifiers;
        private double[] errorRates;

        public void setIterationNumber(int iterationNumber) {
            this.iterationNumber = iterationNumber;
        }

        public int getIterationNumber() {
            return iterationNumber;
        }

        public void setClassifiers(List<TrainableClassifier<T, S>> classifiers) {
            this.classifiers = classifiers;
        }

        public List<TrainableClassifier<T, S>> getClassifiers() {
            return classifiers;
        }

        public void setErrorRates(double[] errorRates) {
            this.errorRates = errorRates;
        }

        public double[] getErrorRates() {
            return errorRates;
        }
    }

    public enum CoTrainingMethod {
        CO_TRAINING {
            @Override
            protected <T extends Vector, S> void updatePredictions(
                    List<Future<DataSet<PredictedDataInstance<T, S>>>> predictionResults,
                    CoTrainingIntegrator<T, S> integrator
            ) throws ExecutionException, InterruptedException {
                for (int i = 0; i < predictionResults.size(); i++) {
                    DataSet<PredictedDataInstance<T, S>> dataSet = predictionResults.get(i).get();
                    for (int j = 0; j < dataSet.size(); j++) {
                        // Keep the highest probability prediction / Most confident prediction
                        PredictedDataInstance<T, S> predictedDataInstance = dataSet.get(j);
                        MultiViewPredictedDataInstance<T, S> unlabeledDataInstance = integrator.unlabeledDataSet.get(j);
                        if (i == 0 || predictedDataInstance.probability() > unlabeledDataInstance.probability()) {
                            integrator.unlabeledDataSet.set(j, new MultiViewPredictedDataInstance<>(
                                                                    unlabeledDataInstance.name(),
                                                                    unlabeledDataInstance.features(),
                                                                    predictedDataInstance.label(),
                                                                    unlabeledDataInstance.source(),
                                                                    predictedDataInstance.probability())
                            );
                        }
                    }
                }
            }
        },
        ROBUST_CO_TRAINING {
            @Override
            protected <T extends Vector, S> void updatePredictions(
                    List<Future<DataSet<PredictedDataInstance<T, S>>>> predictionResults,
                    CoTrainingIntegrator<T, S> integrator
            ) throws ExecutionException, InterruptedException {
                Label label = new Label("label");
                List<Integrator.Data.PredictedInstance> predictedInstances = new ArrayList<>();
                for (int i = 0; i < predictionResults.size(); i++) {
                    DataSet<PredictedDataInstance<T, S>> dataSet = predictionResults.get(i).get();
                    for (int j = 0; j < dataSet.size(); j++)
                        predictedInstances.add(new Integrator.Data.PredictedInstance(
                                j,
                                label,
                                i,
                                dataSet.get(j).label().equals(1.0) ? 1.0 : 0.0)
                        );
                }
                Integrator errorEstimation =
                        new AgreementIntegrator.Builder(new Integrator.Data<>(predictedInstances))
                                .highestOrder(4)
                                .onlyEvenCardinalitySubsetsAgreements(false)
                                .build();
                integrator.errorRates = new double[integrator.errorRates.length];
                errorEstimation.errorRates().stream()
                        .forEach(errorRate -> integrator.errorRates[errorRate.functionId()] = errorRate.value());
                for (int i = 0; i < predictionResults.size(); i++) {
                    DataSet<PredictedDataInstance<T, S>> dataSet = predictionResults.get(i).get();
                    for (int j = 0; j < dataSet.size(); j++) {
                        // Keep the highest probability prediction / Most confident prediction
                        PredictedDataInstance<T, S> predictedDataInstance = dataSet.get(j);
                        MultiViewPredictedDataInstance<T, S> unlabeledDataInstance = integrator.unlabeledDataSet.get(j);
                        double weightedProbability = predictedDataInstance.probability() * (1 - integrator.errorRates[i]);
                        if (i == 0 || weightedProbability > unlabeledDataInstance.probability()) {
                            integrator.unlabeledDataSet.set(j, new MultiViewPredictedDataInstance<>(
                                    unlabeledDataInstance.name(),
                                    unlabeledDataInstance.features(),
                                    predictedDataInstance.label(),
                                    unlabeledDataInstance.source(),
                                    weightedProbability)
                            );
                        }
                    }
                }
            }
        },
        ROBUST_CO_TRAINING_BEE {
            @Override
            protected <T extends Vector, S> void updatePredictions(
                    List<Future<DataSet<PredictedDataInstance<T, S>>>> predictionResults,
                    CoTrainingIntegrator<T, S> integrator
            ) throws ExecutionException, InterruptedException {
                Label label = new Label("label");
                List<Integrator.Data.PredictedInstance> predictedInstances = new ArrayList<>();
                for (int i = 0; i < predictionResults.size(); i++) {
                    DataSet<PredictedDataInstance<T, S>> dataSet = predictionResults.get(i).get();
                    for (int j = 0; j < dataSet.size(); j++)
                        predictedInstances.add(new Integrator.Data.PredictedInstance(
                                j,
                                label,
                                i,
                                dataSet.get(j).label().equals(1.0) ? 1.0 : 0.0)
                        );
                }
                Integrator errorEstimation = new BayesianIntegrator.Builder(new Integrator.Data<>(predictedInstances)).build();
                integrator.errorRates = new double[integrator.errorRates.length];
                errorEstimation.errorRates().stream()
                        .forEach(errorRate -> integrator.errorRates[errorRate.functionId()] = errorRate.value());
                for (int i = 0; i < predictionResults.size(); i++) {
                    DataSet<PredictedDataInstance<T, S>> dataSet = predictionResults.get(i).get();
                    for (int j = 0; j < dataSet.size(); j++) {
                        // Keep the highest probability prediction / Most confident prediction
                        PredictedDataInstance<T, S> predictedDataInstance = dataSet.get(j);
                        MultiViewPredictedDataInstance<T, S> unlabeledDataInstance = integrator.unlabeledDataSet.get(j);
                        double weightedProbability = predictedDataInstance.probability() * (1 - integrator.errorRates[i]);
                        if (i == 0 || weightedProbability > unlabeledDataInstance.probability()) {
                            integrator.unlabeledDataSet.set(j, new MultiViewPredictedDataInstance<>(
                                    unlabeledDataInstance.name(),
                                    unlabeledDataInstance.features(),
                                    predictedDataInstance.label(),
                                    unlabeledDataInstance.source(),
                                    weightedProbability)
                            );
                        }
                    }
                }
            }
        },
        TRUE_ERRORS_ROBUST_CO_TRAINING {
            @Override
            protected <T extends Vector, S> void updatePredictions(
                    List<Future<DataSet<PredictedDataInstance<T, S>>>> predictionResults,
                    CoTrainingIntegrator<T, S> integrator
            ) throws ExecutionException, InterruptedException {
                List<boolean[]> classifierOutputs = new ArrayList<>();
                integrator.errorRates = new double[predictionResults.size()];
                for (int i = 0; i < predictionResults.size(); i++) {
                    DataSet<PredictedDataInstance<T, S>> dataSet = predictionResults.get(i).get();
                    integrator.errorRates[i] = 0;
                    for (int j = 0; j < dataSet.size(); j++) {
                        if (i == 0)
                            classifierOutputs.add(new boolean[predictionResults.size()]);
                        classifierOutputs.get(j)[i] = dataSet.get(j).label().equals(1.0);
                        integrator.errorRates[i] += dataSet.get(j).label().equals(integrator.trueLabels.get(dataSet.get(j).name())) ? 0 : 1;
                    }
                    integrator.errorRates[i] /= dataSet.size();
                }
                for (int i = 0; i < predictionResults.size(); i++) {
                    DataSet<PredictedDataInstance<T, S>> dataSet = predictionResults.get(i).get();
                    for (int j = 0; j < dataSet.size(); j++) {
                        // Keep the highest probability prediction / Most confident prediction
                        PredictedDataInstance<T, S> predictedDataInstance = dataSet.get(j);
                        MultiViewPredictedDataInstance<T, S> unlabeledDataInstance = integrator.unlabeledDataSet.get(j);
                        double weightedProbability = predictedDataInstance.probability() * (1 - integrator.errorRates[i]);
                        if (i == 0 || weightedProbability > unlabeledDataInstance.probability()) {
                            integrator.unlabeledDataSet.set(j, new MultiViewPredictedDataInstance<>(
                                                                    unlabeledDataInstance.name(),
                                                                    unlabeledDataInstance.features(),
                                                                    predictedDataInstance.label(),
                                                                    unlabeledDataInstance.source(),
                                                                    weightedProbability)
                            );
                        }
                    }
                }
            }
        };

        protected abstract <T extends Vector, S> void updatePredictions(
                List<Future<DataSet<PredictedDataInstance<T, S>>>> predictionResults,
                CoTrainingIntegrator<T, S> integrator
        ) throws ExecutionException, InterruptedException;
    }

    public enum DataSelectionMethod {
        FIXED_PROPORTION {
            @Override
            protected <T extends Vector, S> void transferData(
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
            protected <T extends Vector, S> void transferData(
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

        protected abstract <T extends Vector, S> void transferData(
                MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> labeledDataSet,
                MultiViewDataSet<MultiViewPredictedDataInstance<T, S>> unlabeledDataSet,
                double parameter
        );
    }
}
