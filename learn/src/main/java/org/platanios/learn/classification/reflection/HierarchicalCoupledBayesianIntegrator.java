package org.platanios.learn.classification.reflection;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.commons.math3.random.RandomDataGenerator;
import org.platanios.learn.classification.Label;
import org.platanios.utilities.ArrayUtilities;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public final class HierarchicalCoupledBayesianIntegrator extends Integrator {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final BiMap<Label, Integer> labelKeysMap = HashBiMap.create();
    private final BiMap<Integer, Integer> classifierKeysMap = HashBiMap.create();

    private final List<BiMap<Integer, Integer>> instanceKeysMap;
    private final int[][] domainInstances;
    private final int[][] domainFunctions;
    private final double[][] domainValues;
    private final int[][][] predictionsByFunctionInstances;
    private final double[][][] predictionsByFunctionValues;
    private final int[][][] predictionsByInstanceFunctions;
    private final double[][][] predictionsByInstanceValues;
    private final double labelsPriorAlpha;
    private final double labelsPriorBeta;
    private final double errorRatesPriorAlpha;
    private final double errorRatesPriorBeta;
    private final FastHDPPrior hdp;
    private final int numberOfBurnInSamples;
    private final int numberOfThinningSamples;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int maximumNumberOfClusters;
    private final int[] numberOfInstances;
    private final double[][] labelPriorsSamples;      // indexed by sample, domain
    private final double[][] labelPriorsCounts;       // indexed by domain, 0/1
    private final double[][] errorRatesSamples;
    private final double[][] errorRatesCounts;
    private final int[][][] clusterAssignmentSamples;
    private final int[][][] labelsSamples;
    private final double[] labelPriorMeans;
    private final double[][] errorRateMeans;
    private final double[][] labelMeans;

    private boolean needsInference = true;

    protected static abstract class AbstractBuilder<T extends AbstractBuilder<T>>
            extends Integrator.AbstractBuilder<T> {
        private double labelsPriorAlpha = 1.0;
        private double labelsPriorBeta = 1.0;
        private double errorRatesPriorAlpha = 1.0;
        private double errorRatesPriorBeta = 2.0;
        private double alpha = 1.0;
        private double gamma = 1.0;
        private int numberOfBurnInSamples = 4000;
        private int numberOfThinningSamples = 10;
        private int numberOfSamples = 200;

        public AbstractBuilder(Integrator.Data<Integrator.Data.PredictedInstance> data) {
            super(data);
        }

        public T labelsPriorAlpha(double labelsPriorAlpha) {
            this.labelsPriorAlpha = labelsPriorAlpha;
            return self();
        }

        public T labelsPriorBeta(double labelsPriorBeta) {
            this.labelsPriorBeta = labelsPriorBeta;
            return self();
        }

        public T errorRatesPriorAlpha(double errorRatesPriorAlpha) {
            this.errorRatesPriorAlpha = errorRatesPriorAlpha;
            return self();
        }

        public T errorRatesPriorBeta(double errorRatesPriorBeta) {
            this.labelsPriorAlpha = labelsPriorAlpha;
            return self();
        }

        public T alpha(double alpha) {
            this.alpha = alpha;
            return self();
        }

        public T gamma(double gamma) {
            this.gamma = gamma;
            return self();
        }

        public T numberOfBurnInSamples(int numberOfBurnInSamples) {
            this.numberOfBurnInSamples = numberOfBurnInSamples;
            return self();
        }

        public T numberOfThinningSamples(int numberOfThinningSamples) {
            this.numberOfThinningSamples = numberOfThinningSamples;
            return self();
        }

        public T numberOfSamples(int numberOfSamples) {
            this.numberOfSamples = numberOfSamples;
            return self();
        }

        public HierarchicalCoupledBayesianIntegrator build() {
            return new HierarchicalCoupledBayesianIntegrator(this);
        }
    }

    public static class Builder extends AbstractBuilder<Builder> {
        public Builder(Integrator.Data<Integrator.Data.PredictedInstance> data) {
            super(data);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }

    private HierarchicalCoupledBayesianIntegrator(AbstractBuilder<?> builder) {
        super(builder);
        data.stream()
                .map(Integrator.Data.PredictedInstance::label)
                .distinct()
                .forEach(label -> labelKeysMap.computeIfAbsent(label, key -> labelKeysMap.size()));
        labelsPriorAlpha = builder.labelsPriorAlpha;
        labelsPriorBeta = builder.labelsPriorBeta;
        errorRatesPriorAlpha = builder.errorRatesPriorAlpha;
        errorRatesPriorBeta = builder.errorRatesPriorBeta;
        numberOfBurnInSamples = builder.numberOfBurnInSamples;
        numberOfThinningSamples = builder.numberOfThinningSamples;
        numberOfSamples = builder.numberOfSamples;
        numberOfFunctions = (int) data.stream().map(Integrator.Data.PredictedInstance::functionId).distinct().count();
        numberOfDomains = (int) data.stream().map(Integrator.Data.PredictedInstance::label).distinct().count();
        maximumNumberOfClusters = numberOfDomains * numberOfFunctions + numberOfFunctions;
        hdp = new FastHDPPrior(numberOfDomains, numberOfFunctions, builder.alpha, builder.gamma);
        instanceKeysMap = new ArrayList<>();
        numberOfInstances = new int[numberOfDomains];
        domainInstances = new int[numberOfDomains][];
        domainFunctions = new int[numberOfDomains][];
        domainValues = new double[numberOfDomains][];
        predictionsByFunctionInstances = new int[numberOfDomains][numberOfFunctions][];
        predictionsByFunctionValues = new double[numberOfDomains][numberOfFunctions][];
        predictionsByInstanceFunctions = new int[numberOfDomains][][];
        predictionsByInstanceValues = new double[numberOfDomains][][];
        for (int p = 0; p < numberOfDomains; p++) {
            final int domain = p;
            instanceKeysMap.add(HashBiMap.create());
            int numberOfSamples = (int) data.stream().filter(instance -> instance.label().equals(labelKeysMap.inverse().get(domain))).count();
            domainInstances[p] = new int[numberOfSamples];
            domainFunctions[p] = new int[numberOfSamples];
            domainValues[p] = new double[numberOfSamples];
            numberOfInstances[p] = (int) data.stream().filter(instance -> instance.label().equals(labelKeysMap.inverse().get(domain))).map(Data.PredictedInstance::id).distinct().count();
            int[] numberOfClassifiersPerInstance = new int[numberOfInstances[p]];
            int[] sampleIndex = {0};
            data.stream().filter(instance -> instance.label().equals(labelKeysMap.inverse().get(domain))).forEach(instance -> {
                int i = instanceKeysMap.get(domain).computeIfAbsent(instance.id(), key -> instanceKeysMap.get(domain).size());
                int j = classifierKeysMap.computeIfAbsent(instance.functionId(), key -> classifierKeysMap.size());
                double value = instance.value() >= 0.5 ? 1.0 : 0.0;
                domainInstances[domain][sampleIndex[0]] = i;
                domainFunctions[domain][sampleIndex[0]] = j;
                domainValues[domain][sampleIndex[0]] = value;
                numberOfClassifiersPerInstance[i]++;
                sampleIndex[0]++;
            });
            int[][] byFunctionSampleIndex = new int[numberOfFunctions][];
            for (int j = 0; j < numberOfFunctions; j++) {
                byFunctionSampleIndex[j] = new int[] {0};
                final int functionIndex = j;
                int count =
                        (int) data.stream()
                                .filter(instance -> instance.label().equals(labelKeysMap.inverse().get(domain))
                                        && instance.functionId() == classifierKeysMap.inverse().get(functionIndex))
                                .count();
                predictionsByFunctionInstances[p][j] = new int[count];
                predictionsByFunctionValues[p][j] = new double[count];
            }
            predictionsByInstanceFunctions[p] = new int[numberOfInstances[p]][];
            predictionsByInstanceValues[p] = new double[numberOfInstances[p]][];
            int[][] byInstanceSampleIndex = new int[numberOfInstances[p]][];
            for (int i = 0; i < numberOfInstances[p]; i++) {
                byInstanceSampleIndex[i] = new int[] {0};
                predictionsByInstanceFunctions[p][i] = new int[numberOfClassifiersPerInstance[i]];
                predictionsByInstanceValues[p][i] = new double[numberOfClassifiersPerInstance[i]];
            }
            data.stream().filter(instance -> instance.label().equals(labelKeysMap.inverse().get(domain))).forEach(instance -> {
                int i = instanceKeysMap.get(domain).get(instance.id());
                int j = classifierKeysMap.get(instance.functionId());
                double value = instance.value() >= 0.5 ? 1.0 : 0.0;
                predictionsByFunctionInstances[domain][j][byFunctionSampleIndex[j][0]] = i;
                predictionsByFunctionValues[domain][j][byFunctionSampleIndex[j][0]] = value;
                predictionsByInstanceFunctions[domain][i][byInstanceSampleIndex[i][0]] = j;
                predictionsByInstanceValues[domain][i][byInstanceSampleIndex[i][0]] = value;
                byFunctionSampleIndex[j][0]++;
                byInstanceSampleIndex[i][0]++;
            });
        }
        labelPriorsSamples = new double[numberOfSamples][numberOfDomains];
        labelPriorsCounts = new double[numberOfDomains][2];
        errorRatesSamples = new double[numberOfSamples][maximumNumberOfClusters];
        errorRatesCounts = new double[maximumNumberOfClusters][2];
        clusterAssignmentSamples = new int[numberOfSamples][numberOfDomains][numberOfFunctions];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];
        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
            for (int p = 0; p < numberOfDomains; p++)
                labelsSamples[sampleIndex][p] = new int[numberOfInstances[p]];
        for (int p = 0; p < numberOfDomains; p++) {
            int[] sum = new int[numberOfInstances[p]];
            int[] numberOfFunctions = new int[numberOfInstances[p]];
            for (int sample = 0; sample < domainInstances[p].length; sample++) {
                sum[domainInstances[p][sample]] += domainValues[p][sample];
                numberOfFunctions[domainInstances[p][sample]]++;
            }
            for (int i = 0; i < numberOfInstances[p]; i++) {
                labelsSamples[0][p][i] = sum[i] >= (numberOfFunctions[i] / 2) ? 1 : 0;
                updateCountsAfterSamplingLabel(0, p, i);
            }
            for (int j = 0; j < this.numberOfFunctions; j++) {
                clusterAssignmentSamples[0][p][j] = 0;
                hdp.add_items_table_assignment(p, j, 0, 0);
            }
        }
        samplePriorsAndErrorRates(0);
        labelPriorMeans = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
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
        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
            samplePriorsAndErrorRates(0);
            sampleClusterAssignments(0);
            sampleLabels(0);
        }
        for (int sampleIndex = 1; sampleIndex < numberOfSamples; sampleIndex++) {
            for (int i = 0; i < numberOfThinningSamples + 1; i++) {
                samplePriorsAndErrorRates(sampleIndex - 1);
                sampleClusterAssignments(sampleIndex - 1);
                sampleLabels(sampleIndex - 1);
            }
            storeSample(sampleIndex);
        }
        // Aggregate values for means and variances computation
        for (int p = 0; p < numberOfDomains; p++)
            labelMeans[p] = new double[numberOfInstances[p]];
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                labelPriorMeans[p] += labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]];
                for (int i = 0; i < numberOfInstances[p]; i++)
                    labelMeans[p][i] += labelsSamples[sampleNumber][p][i];
            }
        }
        List<Integrator.Data.PredictedInstance> integratedDataInstances = new ArrayList<>();
        List<ErrorRates.Instance> errorRatesInstances = new ArrayList<>();
        // Compute values for the means and the variances
        for (int p = 0; p < numberOfDomains; p++) {
            labelPriorMeans[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++) {
                errorRateMeans[p][j] /= numberOfSamples;
                errorRatesInstances.add(new ErrorRates.Instance(labelKeysMap.inverse().get(p),
                                                                classifierKeysMap.inverse().get(j),
                                                                errorRateMeans[p][j]));
            }
            for (int i = 0; i < numberOfInstances[p]; i++) {
                labelMeans[p][i] /= numberOfSamples;
                integratedDataInstances.add(new Data.PredictedInstance(instanceKeysMap.get(p).inverse().get(i),
                                                                       labelKeysMap.inverse().get(p),
                                                                       labelMeans[p][i]));
            }
        }
        integratedData = new Data<>(integratedDataInstances);
        errorRates = new ErrorRates(errorRatesInstances);
        needsInference = false;
    }

    private void samplePriorsAndErrorRates(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++)
            labelPriorsSamples[sampleNumber][p] = randomDataGenerator.nextBeta(
                    labelsPriorAlpha + labelPriorsCounts[p][1],
                    labelsPriorBeta + labelPriorsCounts[p][0]
            );
        for (int k = 0; k < maximumNumberOfClusters; k++) {
            errorRatesSamples[sampleNumber][k] = randomDataGenerator.nextBeta(
                    errorRatesPriorAlpha + errorRatesCounts[k][0],
                    errorRatesPriorBeta + errorRatesCounts[k][1]
            );
        }
    }

    private void updateCountsBeforeSamplingClusterAssignment(int sampleNumber, int p, int j) {
        for (int sample = 0; sample < predictionsByFunctionInstances[p][j].length; sample++)
            if (labelsSamples[sampleNumber][p][predictionsByFunctionInstances[p][j][sample]] != predictionsByFunctionValues[p][j][sample])
                errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][0]--;
            else
                errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][1]--;
        hdp.remove_items_table_assignment(p, j);
    }

    private void updateCountsAfterSamplingClusterAssignment(int sampleNumber, int p, int j, int tableId) {
        for (int sample = 0; sample < predictionsByFunctionInstances[p][j].length; sample++)
            if (labelsSamples[sampleNumber][p][predictionsByFunctionInstances[p][j][sample]] != predictionsByFunctionValues[p][j][sample])
                errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][0]++;
            else
                errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][1]++;
        hdp.add_items_table_assignment(p, j, tableId, clusterAssignmentSamples[sampleNumber][p][j]);
    }

    private void sampleClusterAssignments(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                updateCountsBeforeSamplingClusterAssignment(sampleNumber, p, j);
                int currentNumberOfClusters = hdp.prob_table_assignment_for_item(p, j);
                double cdf[] = new double[currentNumberOfClusters];
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < currentNumberOfClusters; k++) {
                    int clusterID = hdp.pdf[k].topic;
                    cdf[k] = Math.log(hdp.pdf[k].prob);
                    cdf[k] += errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][0] * Math.log(errorRatesSamples[sampleNumber][clusterID]);
                    cdf[k] += errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][j]][1] * Math.log(1 - errorRatesSamples[sampleNumber][clusterID]);
                    if (max < cdf[k])
                        max = cdf[k];
                }
                cdf[0] -= max;
                for (int k = 1; k < currentNumberOfClusters; k++) {
                    cdf[k] -= max;
                    cdf[k] = Math.log(Math.exp(cdf[k - 1]) + Math.exp(cdf[k]));
                }
                double uniform = Math.log(random.nextDouble()) + cdf[currentNumberOfClusters - 1];
                int newClusterID = hdp.pdf[currentNumberOfClusters - 1].topic;
                int newTableID = hdp.pdf[currentNumberOfClusters - 1].table;
                clusterAssignmentSamples[sampleNumber][p][j] = newClusterID;
                for (int k = 0; k < currentNumberOfClusters - 1; k++) {
                    if (cdf[k] > uniform) {
                    	newClusterID = hdp.pdf[k].topic;
                    	newTableID = hdp.pdf[k].table;
                        clusterAssignmentSamples[sampleNumber][p][j] = newClusterID;
                        break;
                    }
                }
                updateCountsAfterSamplingClusterAssignment(sampleNumber, p, j,newTableID);
            }
        }
    }

    private void updateCountsBeforeSamplingLabel(int sampleNumber, int p, int i) {
        labelPriorsCounts[p][labelsSamples[sampleNumber][p][i]]--;
        for (int sample = 0; sample < predictionsByInstanceFunctions[p][i].length; sample++)
            if (labelsSamples[sampleNumber][p][i] != predictionsByInstanceValues[p][i][sample])
                errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][predictionsByInstanceFunctions[p][i][sample]]][0]--;
            else
                errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][predictionsByInstanceFunctions[p][i][sample]]][1]--;
    }

    private void updateCountsAfterSamplingLabel(int sampleNumber, int p, int i) {
        labelPriorsCounts[p][labelsSamples[sampleNumber][p][i]]++;
        for (int sample = 0; sample < predictionsByInstanceFunctions[p][i].length; sample++)
            if (labelsSamples[sampleNumber][p][i] != predictionsByInstanceValues[p][i][sample])
                errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][predictionsByInstanceFunctions[p][i][sample]]][0]++;
            else
                errorRatesCounts[clusterAssignmentSamples[sampleNumber][p][predictionsByInstanceFunctions[p][i][sample]]][1]++;
    }

    private void sampleLabels(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            double[] p0 = new double[numberOfInstances[p]];
            double[] p1 = new double[numberOfInstances[p]];
            Arrays.fill(p0, 1 - labelPriorsSamples[sampleNumber][p]);
            Arrays.fill(p1, labelPriorsSamples[sampleNumber][p]);
            for (int sample = 0; sample < domainInstances[p].length; sample++)
                if (domainValues[p][sample] == 0.0) {
                    p0[domainInstances[p][sample]] *= (1.0 - errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][domainFunctions[p][sample]]]);
                    p1[domainInstances[p][sample]] *= errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][domainFunctions[p][sample]]];
                } else {
                    p0[domainInstances[p][sample]] *= errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][domainFunctions[p][sample]]];
                    p1[domainInstances[p][sample]] *= (1.0 - errorRatesSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][domainFunctions[p][sample]]]);
                }
            for (int i = 0; i < numberOfInstances[p]; i++) {
                int newLabel = randomDataGenerator.nextBinomial(1, p1[i] / (p0[i] + p1[i]));
                if (labelsSamples[sampleNumber][p][i] != newLabel) {
                    updateCountsBeforeSamplingLabel(sampleNumber, p, i);
                    labelsSamples[sampleNumber][p][i] = newLabel;
                    updateCountsAfterSamplingLabel(sampleNumber, p, i);
                }
            }
        }
    }

    private void storeSample(int sampleIndex) {
        ArrayUtilities.copy(labelPriorsSamples[sampleIndex - 1], labelPriorsSamples[sampleIndex]);
        ArrayUtilities.copy(errorRatesSamples[sampleIndex - 1], errorRatesSamples[sampleIndex]);
        ArrayUtilities.copy(clusterAssignmentSamples[sampleIndex - 1], clusterAssignmentSamples[sampleIndex]);
        ArrayUtilities.copy(labelsSamples[sampleIndex - 1], labelsSamples[sampleIndex]);
    }

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
//        // Top level HDP term
//        for (int topicID : hdp.getTakenTopics().toArray())
//            logLikelihood += Math.log(hdp.getNumberOfTablesForTopic(topicID)) - Math.log(hdp.getNumberOfTables());
//        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
//            sampleLabels(sampleNumber);
//            for (int p = 0; p < numberOfDomains; p++) {
//                // Label prior term
//                logLikelihood += (labelsPriorAlpha - 1) * Math.log(labelPriorsSamples[sampleNumber][p])
//                        + (labelsPriorBeta - 1) * Math.log(1 - labelPriorsSamples[sampleNumber][p]);
//                // Bottom level HDP term
//                Map<Integer, AtomicInteger> clusterCounts = new HashMap<>();
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    if (!clusterCounts.containsKey(clusterAssignmentSamples[sampleNumber][p][j]))
//                        clusterCounts.put(clusterAssignmentSamples[sampleNumber][p][j], new AtomicInteger(1));
//                    else
//                        clusterCounts.get(clusterAssignmentSamples[sampleNumber][p][j]).incrementAndGet();
//                }
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
//                    logLikelihood += confusionMatrixPrior[0][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][0][0]);
//                    logLikelihood += confusionMatrixPrior[0][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][0][1]);
//                    logLikelihood += confusionMatrixPrior[1][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][1][0]);
//                    logLikelihood += confusionMatrixPrior[1][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][1][1]);
//                }
//                // Function outputs term
//                for (int j = 0; j < numberOfFunctions; j++)
//                    for (int i = 0; i < numberOfInstances[p]; i++)
//                        logLikelihood += Math.log(confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]);
//            }
//        }
//        return logLikelihood / numberOfSamples;
//    }
}
