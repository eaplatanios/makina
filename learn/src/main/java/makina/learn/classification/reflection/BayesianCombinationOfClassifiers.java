//package makina.learn.classification.reflection;
//
//import com.google.common.collect.BiMap;
//import com.google.common.collect.HashBiMap;
//import makina.learn.classification.Label;
//import org.apache.commons.math3.random.RandomDataGenerator;
//
//import java.lang.reflect.Array;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//import java.util.Random;
//import java.util.concurrent.atomic.AtomicInteger;
//
///**
// * @author Emmanouil Antonios Platanios
// */
//public final class BayesianCombinationOfClassifiers extends Integrator {
//    private final Random random = new Random();
//    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
//    private final BiMap<Label, Integer> labelKeysMap = HashBiMap.create();
//    private final BiMap<Integer, Integer> classifierKeysMap = HashBiMap.create();
//
//    private final List<BiMap<Integer, Integer>> instanceKeysMap;
//    private final int[][] domainInstances;
//    private final int[][] domainFunctions;
//    private final double[][] domainValues;
//    private final double labelsPriorAlpha;
//    private final double labelsPriorBeta;
//    private final double confusionMatrixPrior;
//    private final DirichletProcess dpPriors[];
//    private final int numberOfBurnInSamples;
//    private final int numberOfThinningSamples;
//    private final int numberOfSamples;
//    private final int numberOfFunctions;
//    private final int numberOfDomains;
//    private final int[] numberOfInstances;
//    private int[][][] labelsSamples;
//    private int[][][] functionOutputsArray;
//    private final double[][] labelPriorsSamples;      // indexed by sample, domain
//    private final double[][] labelPriorsCounts;       // indexed by domain, 0/1
//    private final double[][][][][] confusionMatrixSamples;  // indexed by sample, domain, cluster id, 0/1, 0/1
//    private final double[][][][] confusionMatrixCounts;     // indexed by domain, cluster id, 0/1, 0/1
//    private final int[][][] clusterAssignmentSamples;
//    private double[][] errorRateMeans;
//    private double[] labelPriorMeans;
//    private double[][] labelMeans;
//
//    protected static abstract class AbstractBuilder<T extends BayesianCombinationOfClassifiers.AbstractBuilder<T>>
//            extends Integrator.AbstractBuilder<T> {
//        private Long seed = null;
//        private double labelsPriorAlpha = 1.0;
//        private double labelsPriorBeta = 1.0;
//        private double[][] confusionMatrixPrior = new double[][] { new double[] { 5, 0.5 }, new double[] { 0.5, 5 } };
//        private double alpha = 1.0;
//        private int numberOfBurnInSamples = 4000;
//        private int numberOfThinningSamples = 10;
//        private int numberOfSamples = 200;
//
//        public AbstractBuilder(Integrator.Data<Integrator.Data.PredictedInstance> data) {
//            super(data);
//        }
//
//        private AbstractBuilder(String predictedDataFilename) {
//            super(predictedDataFilename);
//        }
//
//        public T seed(Long seed) {
//            this.seed = seed;
//            return self();
//        }
//
//        public T labelsPriorAlpha(double labelsPriorAlpha) {
//            this.labelsPriorAlpha = labelsPriorAlpha;
//            return self();
//        }
//
//        public T labelsPriorBeta(double labelsPriorBeta) {
//            this.labelsPriorBeta = labelsPriorBeta;
//            return self();
//        }
//
//        public T confusionMatrixPrior(double[][] confusionMatrixPrior) {
//            this.confusionMatrixPrior = confusionMatrixPrior;
//            return self();
//        }
//
//        public T alpha(double alpha) {
//            this.alpha = alpha;
//            return self();
//        }
//
//        public T numberOfBurnInSamples(int numberOfBurnInSamples) {
//            this.numberOfBurnInSamples = numberOfBurnInSamples;
//            return self();
//        }
//
//        public T numberOfThinningSamples(int numberOfThinningSamples) {
//            this.numberOfThinningSamples = numberOfThinningSamples;
//            return self();
//        }
//
//        public T numberOfSamples(int numberOfSamples) {
//            this.numberOfSamples = numberOfSamples;
//            return self();
//        }
//
//        public T options(String options) {
//            if (options == null || options.length() == 0)
//                return self();
//            if (options.length() < 8)
//                throw new IllegalArgumentException("Invalid number of arguments provided.");
//            String[] optionsStringParts = options.split(":");
//            if (!optionsStringParts[0].equals("-"))
//                this.numberOfBurnInSamples = Integer.parseInt(optionsStringParts[0]);
//            if (!optionsStringParts[1].equals("-"))
//                this.numberOfThinningSamples = Integer.parseInt(optionsStringParts[1]);
//            if (!optionsStringParts[2].equals("-"))
//                this.numberOfSamples = Integer.parseInt(optionsStringParts[2]);
//            if (!optionsStringParts[3].equals("-"))
//                this.alpha = Double.parseDouble(optionsStringParts[3]);
//            if (!optionsStringParts[4].equals("-"))
//                this.labelsPriorAlpha = Double.parseDouble(optionsStringParts[4]);
//            if (!optionsStringParts[5].equals("-"))
//                this.labelsPriorBeta = Double.parseDouble(optionsStringParts[5]);
//            if (!optionsStringParts[6].equals("-")) {
//                String[] matrix = optionsStringParts[6].split(",");
//                this.confusionMatrixPrior = new double[][]{
//                        new double[]{Double.parseDouble(matrix[0]), Double.parseDouble(matrix[1])},
//                        new double[]{Double.parseDouble(matrix[2]), Double.parseDouble(matrix[3])}};
//            }
//            return self();
//        }
//
//        public BayesianCombinationOfClassifiers build() {
//            return new BayesianCombinationOfClassifiers(this);
//        }
//    }
//
//    public static class Builder
//            extends BayesianCombinationOfClassifiers.AbstractBuilder<BayesianCombinationOfClassifiers.Builder> {
//        public Builder(Integrator.Data<Integrator.Data.PredictedInstance> data) {
//            super(data);
//        }
//
//        public Builder(String predictedDataFilename) {
//            super(predictedDataFilename);
//        }
//
//        @Override
//        protected BayesianCombinationOfClassifiers.Builder self() {
//            return this;
//        }
//    }
//
//    private BayesianCombinationOfClassifiers(AbstractBuilder<?> builder) {
//        super(builder);
//        if (builder.seed != null)
//            randomDataGenerator.reSeed(builder.seed);
//        data.stream()
//                .map(Integrator.Data.PredictedInstance::label)
//                .distinct()
//                .forEach(label -> labelKeysMap.computeIfAbsent(label, key -> labelKeysMap.size()));
//        labelsPriorAlpha = builder.labelsPriorAlpha;
//        labelsPriorBeta = builder.labelsPriorBeta;
//
//
//
//
//
//        numberOfFunctions = functionOutputs.get(0)[0].length;
//        numberOfDomains = functionOutputs.size();
//        numberOfInstances = new int[numberOfDomains];
//        maximumNumberOfClusters = numberOfFunctions + 1;
//        functionOutputsArray = new int[numberOfFunctions][numberOfDomains][];
//        for (int p = 0; p < numberOfDomains; p++) {
//            numberOfInstances[p] = functionOutputs.get(p).length;
//            for (int j = 0; j < numberOfFunctions; j++) {
//                functionOutputsArray[j][p] = new int[numberOfInstances[p]];
//                for (int i = 0; i < numberOfInstances[p]; i++)
//                    functionOutputsArray[j][p][i] = functionOutputs.get(p)[i][j] ? 1 : 0;
//            }
//        }
//        dpPriors = new DirichletProcess[numberOfDomains];
//
//        labelsSamples = new int[numberOfSamples][numberOfDomains][];
//        labelPriorsSamples = new double[numberOfSamples][numberOfDomains];
//        labelPriorsCounts = new double[numberOfDomains][2];
//        confusionMatrixSamples = new double[numberOfSamples][numberOfDomains][numberOfFunctions][2][2];
//        confusionMatrixCounts = new double[numberOfDomains][numberOfFunctions][2][2];
//        clusterAssignmentSamples = new int[numberOfSamples][numberOfDomains][numberOfFunctions];
//
//        labelPriorMeans = new double[numberOfDomains];
//        priorVariances = new double[numberOfDomains];
//        labelMeans = new double[numberOfDomains][];
//        labelVariances = new double[numberOfDomains][];
//        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
//        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
//        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
//            for (int p = 0; p < numberOfDomains; p++)
//                labelsSamples[sampleIndex][p] = new int[numberOfInstances[p]];
//        for (int p = 0; p < numberOfDomains; p++) {
//            labelMeans[p] = new double[numberOfInstances[p]];
//            labelVariances[p] = new double[numberOfInstances[p]];
//            dpPriors[p] = new DirichletProcess(alpha, numberOfFunctions);
//            labelPriorsSamples[0][p] = 0.5;
//            for (int j = 0; j < numberOfFunctions; j++) {
////                clusterAssignmentSamples[0][p][j] = 0;
//                clusterAssignmentSamples[0][p][j] = j;
//                dpPriors[p].addMemberToCluster(clusterAssignmentSamples[0][p][j]);
//            }
//            for (int i = 0; i < numberOfInstances[p]; i++) {
//                int sum = 0;
//                for (int j = 0; j < numberOfFunctions; j++)
//                    sum += functionOutputsArray[j][p][i];
//                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
//                updateCountsAfterSamplingLabel(0, p, i);
//            }
//        }
//        samplePriors(0);
//        sampleConfusionMatrix(0);
//    }
//
//    public void runGibbsSampler() {
//        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
//            samplePriors(0);
//            sampleConfusionMatrix(0);
//            sampleClusterAssignments(0);
//            sampleLabels(0);
//        }
//        for (int sampleIndex = 1; sampleIndex < numberOfSamples; sampleIndex++) {
//            for (int i = 0; i < numberOfThinningSamples + 1; i++) {
//                samplePriors(sampleIndex - 1);
//                sampleConfusionMatrix(sampleIndex - 1);
//                sampleClusterAssignments(sampleIndex - 1);
//                sampleLabels(sampleIndex - 1);
//            }
//            storeSample(sampleIndex);
//        }
//        // Aggregate values for means and variances computation
//        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
//            for (int p = 0; p < numberOfDomains; p++) {
//                labelPriorMeans[p] += labelPriorsSamples[sampleNumber][p];
////                for (int j = 0; j < numberOfFunctions; j++) {
////                    double errorRate = 0;
////                    for (int i = 0; i < numberOfInstances[p]; i++)
////                        errorRate += functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i] ? 1 : 0;
////                    errorRateMeans[p][j] += errorRate / numberOfInstances[p];
////                }
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    double confusionMatrixSum = confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][0]
//                            + confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][1]
//                            + confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][0]
//                            + confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][1];
//                    errorRateMeans[p][j] += confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][1] * labelPriorsSamples[sampleNumber][p] / confusionMatrixSum
//                            + confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][0] * (1 - labelPriorsSamples[sampleNumber][p]) / confusionMatrixSum;
//                }
//                for (int i = 0; i < numberOfInstances[p]; i++)
//                    labelMeans[p][i] += labelsSamples[sampleNumber][p][i];
//            }
//        }
//        // Compute values for the means and the variances
//        for (int p = 0; p < numberOfDomains; p++) {
//            labelPriorMeans[p] /= numberOfSamples;
//            for (int j = 0; j < numberOfFunctions; j++)
//                errorRateMeans[p][j] /= numberOfSamples;
//            for (int i = 0; i < numberOfInstances[p]; i++)
//                labelMeans[p][i] /= numberOfSamples;
//            for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
//                double temp = labelPriorsSamples[sampleNumber][p] - labelPriorMeans[p];
//                priorVariances[p] += temp * temp;
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    temp = (confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][1] * labelPriorsSamples[sampleNumber][p]
//                            + confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][0] * (1 - labelPriorsSamples[sampleNumber][p]))
//                            - errorRateMeans[p][j];
//                    errorRateVariances[p][j] += temp * temp;
//                }
//                for (int i = 0; i < numberOfInstances[p]; i++) {
//                    temp = labelsSamples[sampleNumber][p][i] - labelMeans[p][i];
//                    labelVariances[p][i] += temp * temp;
//                }
//            }
//            priorVariances[p] /= numberOfSamples;
//            for (int j = 0; j < numberOfFunctions; j++)
//                errorRateVariances[p][j] /= numberOfSamples;
//            for (int i = 0; i < numberOfInstances[p]; i++)
//                labelVariances[p][i] /= numberOfSamples;
//        }
//    }
//
//    private void samplePriors(int sampleNumber) {
//        for (int p = 0; p < numberOfDomains; p++)
//            labelPriorsSamples[sampleNumber][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelPriorsCounts[p][1], labelsPriorBeta + labelPriorsCounts[p][0]);
//    }
//
//    private void sampleConfusionMatrix(int sampleNumber) { // TODO: Add checking for number of error rates below chance.
//        for (int p = 0; p < numberOfDomains; p++) {
//            for(int j = 0; j < numberOfFunctions; j++) {
//                confusionMatrixSamples[sampleNumber][p][j][0][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[0][0] + confusionMatrixCounts[p][j][0][0]
//                                                                                                        + confusionMatrixPrior[1][1] + confusionMatrixCounts[p][j][1][1],
//                                                                                                confusionMatrixPrior[0][1] + confusionMatrixCounts[p][j][0][1]
//                                                                                                        + confusionMatrixPrior[1][0] + confusionMatrixCounts[p][j][1][0]);
//                confusionMatrixSamples[sampleNumber][p][j][1][0] = 1 - confusionMatrixSamples[sampleNumber][p][j][0][0];
//                confusionMatrixSamples[sampleNumber][p][j][0][1] = 1 - confusionMatrixSamples[sampleNumber][p][j][0][0];
//                confusionMatrixSamples[sampleNumber][p][j][1][1] = 1 - confusionMatrixSamples[sampleNumber][p][j][1][0];
//            }
//        }
//    }
//
//    private void updateCountsBeforeSamplingClusterAssignment(int sampleNumber, int p, int j) {
//        for (int i = 0; i < numberOfInstances[p]; i++)
//            confusionMatrixCounts[p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
//        dpPriors[p].removeMemberFromCluster(clusterAssignmentSamples[sampleNumber][p][j]);
//    }
//
//    private void updateCountsAfterSamplingClusterAssignment(int sampleNumber, int p, int j) {
//        for (int i = 0; i < numberOfInstances[p]; i++)
//            confusionMatrixCounts[p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
//        dpPriors[p].addMemberToCluster(clusterAssignmentSamples[sampleNumber][p][j]);
//    }
//
//    private void sampleClusterAssignments(int sampleNumber) {
//        for (int p = 0; p < numberOfDomains; p++) {
//            for (int j = 0; j < numberOfFunctions; j++) {
//                int cnt_errs[][] = new int[2][2];
//                for (int i = 0; i < numberOfInstances[p]; i++)
//                    cnt_errs[labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
//                updateCountsBeforeSamplingClusterAssignment(sampleNumber, p, j);
//                int currentNumberOfClusters = dpPriors[p].computeClustersDistribution();
//                double cdf[] = new double[currentNumberOfClusters];
//                double max = Double.NEGATIVE_INFINITY;
//                for (int k = 0; k < currentNumberOfClusters; k++) {
//                    int clusterID = dpPriors[p].getClusterID(k);
//                    cdf[k] = Math.log(dpPriors[p].getClusterUnnormalizedProbability(k));
//                    cdf[k] += cnt_errs[0][0] * Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][0][0]);
//                    cdf[k] += cnt_errs[0][1] * Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][0][1]);
//                    cdf[k] += cnt_errs[1][0] * Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][1][0]);
//                    cdf[k] += cnt_errs[1][1] * Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][1][1]);
//                    if (max < cdf[k])
//                        max = cdf[k];
//                }
//                cdf[0] -= max;
//                for (int k = 1; k < currentNumberOfClusters; k++) {
//                    cdf[k] -= max;
//                    cdf[k] = Math.log(Math.exp(cdf[k - 1]) + Math.exp(cdf[k]));
//                }
//                double uniform = Math.log(random.nextDouble()) + cdf[currentNumberOfClusters - 1];
//                int newClusterID = dpPriors[p].getClusterID(currentNumberOfClusters - 1);
//                clusterAssignmentSamples[sampleNumber][p][j] = newClusterID;
//                for (int k = 0; k < currentNumberOfClusters - 1; k++) {
//                    if (cdf[k] > uniform) {
//                        int clusterID = dpPriors[p].getClusterID(k);
//                        clusterAssignmentSamples[sampleNumber][p][j] = clusterID;
//                        break;
//                    }
//                }
//                updateCountsAfterSamplingClusterAssignment(sampleNumber, p, j);
//            }
//        }
//    }
//
//    private void updateCountsBeforeSamplingLabel(int sampleNumber, int p, int i) {
//        labelPriorsCounts[p][labelsSamples[sampleNumber][p][i]]--;
//        for (int j = 0; j < numberOfFunctions; j++)
//            confusionMatrixCounts[p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
//    }
//
//    private void updateCountsAfterSamplingLabel(int sampleNumber, int p, int i) {
//        labelPriorsCounts[p][labelsSamples[sampleNumber][p][i]]++;
//        for (int j = 0; j < numberOfFunctions; j++)
//            confusionMatrixCounts[p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
//    }
//
//    private void sampleLabels(int sampleNumber) {
//        for (int p = 0; p < numberOfDomains; p++) {
//            for (int i = 0; i < numberOfInstances[p]; i++) {
//                double p0 = 1 - labelPriorsSamples[sampleNumber][p]; // TODO: Compute this in log-space
//                double p1 = labelPriorsSamples[sampleNumber][p];
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    p0 *= confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][functionOutputsArray[j][p][i]];
//                    p1 *= confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][functionOutputsArray[j][p][i]];
//                }
//                int newLabel = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
//                if (labelsSamples[sampleNumber][p][i] != newLabel) {
//                    updateCountsBeforeSamplingLabel(sampleNumber, p, i);
//                    labelsSamples[sampleNumber][p][i] = newLabel;
//                    updateCountsAfterSamplingLabel(sampleNumber, p, i);
//                }
//            }
//        }
//    }
//
//    public double logLikelihood(List<boolean[][]> functionOutputs) {
//        numberOfInstances = new int[numberOfDomains];
//        functionOutputsArray = new int[numberOfFunctions][numberOfDomains][];
//        for (int p = 0; p < numberOfDomains; p++) {
//            numberOfInstances[p] = functionOutputs.get(p).length;
//            for (int j = 0; j < numberOfFunctions; j++) {
//                functionOutputsArray[j][p] = new int[numberOfInstances[p]];
//                for (int i = 0; i < numberOfInstances[p]; i++)
//                    functionOutputsArray[j][p][i] = functionOutputs.get(p)[i][j] ? 1 : 0;
//            }
//        }
//        for (int p = 0; p < numberOfDomains; p++) {
//            labelsSamples[0][p] = new int[numberOfInstances[p]];
//            for (int i = 0; i < numberOfInstances[p]; i++) {
//                int sum = 0;
//                for (int j = 0; j < numberOfFunctions; j++)
//                    sum += functionOutputsArray[j][p][i];
//                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
//            }
//        }
//        double logLikelihood = 0;
//        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
//            sampleLabels(sampleNumber);
//            for (int p = 0; p < numberOfDomains; p++) {
//                // Label prior term
//                logLikelihood += (labelsPriorAlpha - 1) * Math.log(labelPriorsSamples[sampleNumber][p])
//                        + (labelsPriorBeta - 1) * Math.log(1 - labelPriorsSamples[sampleNumber][p]);
//                // Cluster assignments term
//                Map<Integer, AtomicInteger> clusterCounts = new HashMap<>();
//                for (int j = 0; j < numberOfFunctions; j++)
//                    if (!clusterCounts.containsKey(clusterAssignmentSamples[sampleNumber][p][j]))
//                        clusterCounts.put(clusterAssignmentSamples[sampleNumber][p][j], new AtomicInteger(1));
//                    else
//                        clusterCounts.get(clusterAssignmentSamples[sampleNumber][p][j]).incrementAndGet();
//                for (int j = 0; j < numberOfFunctions; j++)
//                    logLikelihood += Math.log(clusterCounts.get(clusterAssignmentSamples[sampleNumber][p][j]).intValue()) - Math.log(numberOfFunctions);
//                // Labels term
//                for (int i = 0; i < numberOfInstances[p]; i++) {
//                    if (labelsSamples[sampleNumber][p][i] == 1)
//                        logLikelihood += Math.log(labelPriorsSamples[sampleNumber][p]);
//                    else
//                        logLikelihood += Math.log(1 - labelPriorsSamples[sampleNumber][p]);
//                }
//                // Confusion matrix term
//                for (int clusterID : clusterCounts.keySet()) {
//                    logLikelihood += confusionMatrixPrior[0][0] * Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][0][0]);
//                    logLikelihood += confusionMatrixPrior[0][1] * Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][0][1]);
//                    logLikelihood += confusionMatrixPrior[1][0] * Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][1][0]);
//                    logLikelihood += confusionMatrixPrior[1][1] * Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][1][1]);
//                }
//                // Function outputs term
//                for (int j = 0; j < numberOfFunctions; j++)
//                    for (int i = 0; i < numberOfInstances[p]; i++)
//                        logLikelihood += Math.log(confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]);
//            }
//        }
//        return logLikelihood / numberOfSamples;
//    }
//
//    private void storeSample(int sampleIndex) {
//        copyArray(labelPriorsSamples[sampleIndex - 1], labelPriorsSamples[sampleIndex]);
//        copyArray(confusionMatrixSamples[sampleIndex - 1], confusionMatrixSamples[sampleIndex]);
//        copyArray(clusterAssignmentSamples[sampleIndex - 1], clusterAssignmentSamples[sampleIndex]);
//        copyArray(labelsSamples[sampleIndex - 1], labelsSamples[sampleIndex]);
//    }
//
//    private void copyArray(Object sourceArray, Object destinationArray) {
//        if(sourceArray.getClass().isArray() && destinationArray.getClass().isArray()) {
//            for(int i = 0; i < Array.getLength(sourceArray); i++) {
//                if(Array.get(sourceArray, i) != null && Array.get(sourceArray, i).getClass().isArray())
//                    copyArray(Array.get(sourceArray, i), Array.get(destinationArray, i));
//                else
//                    Array.set(destinationArray, i, Array.get(sourceArray, i));
//            }
//        }
//    }
//
//}