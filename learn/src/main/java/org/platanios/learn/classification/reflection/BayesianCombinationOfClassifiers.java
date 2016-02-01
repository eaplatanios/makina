package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.List;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BayesianCombinationOfClassifiers {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double labelsPriorAlpha = 1;
    private final double labelsPriorBeta = 1;
    private final double[][] confusionMatrixPrior = new double[][] { new double[] { 1, 1 }, new double[] { 1, 1 } };

    private final int numberOfBurnInSamples;
    private final int numberOfThinningSamples;
    private final int numberOfSamples;
    private final double alpha;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfDataSamples;
    private final int[][][] labelsSamples;
    private final int[][][] functionOutputsArray;
    private final double[][] priorSamples;      // indexed by sample, domain
    private final double[][] priorCounts;       // indexed by domain, 0/1
    private final double[][][][][] confusionMatrixSamples;  // indexed by sample, domain, cluster id, 0/1, 0/1
    private final double[][][][] confusionMatrixCounts;     // indexed by domain, cluster id, 0/1, 0/1
    private final int[][][] clusterAssignmentSamples;

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;

    private DirichletProcessPrior dpPriors[];

    public BayesianCombinationOfClassifiers(List<boolean[][]> functionOutputs,
                                            int numberOfBurnInSamples,
                                            int numberOfThinningSamples,
                                            int numberOfSamples,
                                            double alpha) {
        this.numberOfBurnInSamples = numberOfBurnInSamples;
        this.numberOfThinningSamples = numberOfThinningSamples;
        this.numberOfSamples = numberOfSamples;
        this.alpha = alpha;
        numberOfFunctions = functionOutputs.get(0)[0].length;
        numberOfDomains = functionOutputs.size();
        numberOfDataSamples = new int[numberOfDomains];
        functionOutputsArray = new int[numberOfFunctions][numberOfDomains][];
        for (int p = 0; p < numberOfDomains; p++) {
            numberOfDataSamples[p] = functionOutputs.get(p).length;
            for (int j = 0; j < numberOfFunctions; j++) {
                functionOutputsArray[j][p] = new int[numberOfDataSamples[p]];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    functionOutputsArray[j][p][i] = functionOutputs.get(p)[i][j] ? 1 : 0;
            }
        }
        dpPriors = new DirichletProcessPrior[numberOfDomains];

        labelsSamples = new int[numberOfSamples][numberOfDomains][];
        priorSamples = new double[numberOfSamples][numberOfDomains];
        priorCounts = new double[numberOfDomains][2];
        confusionMatrixSamples = new double[numberOfSamples][numberOfDomains][numberOfFunctions][2][2];
        confusionMatrixCounts = new double[numberOfDomains][numberOfFunctions][2][2];
        clusterAssignmentSamples = new int[numberOfSamples][numberOfDomains][numberOfFunctions];

        priorMeans = new double[numberOfDomains];
        priorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
            for (int p = 0; p < numberOfDomains; p++)
                labelsSamples[sampleIndex][p] = new int[numberOfDataSamples[p]];
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfDataSamples[p]];
            labelVariances[p] = new double[numberOfDataSamples[p]];
            dpPriors[p] = new DirichletProcessPrior(alpha, numberOfFunctions);
            priorSamples[0][p] = 0.5;
            for (int j = 0; j < numberOfFunctions; j++) {
                clusterAssignmentSamples[0][p][j] = 0;
                updateCountsAfterSamplingClusterAssignment(0, p, j);
            }
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
                updateCountsAfterSamplingLabel(0, p, i);
            }
        }
        samplePriors(0);
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                confusionMatrixSamples[0][p][0][0][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[0][0] + confusionMatrixCounts[p][0][0][0],
                                                                                     confusionMatrixPrior[0][1] + confusionMatrixCounts[p][0][0][1]);
                confusionMatrixSamples[0][p][0][1][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[1][0] + confusionMatrixCounts[p][0][1][0],
                                                                                     confusionMatrixPrior[1][1] + confusionMatrixCounts[p][0][1][1]);
                confusionMatrixSamples[0][p][0][0][1] = 1 - confusionMatrixSamples[0][p][0][0][0];
                confusionMatrixSamples[0][p][0][1][1] = 1 - confusionMatrixSamples[0][p][0][1][0];
            }
        }

    }

    public void runGibbsSampler() {
        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
            samplePriors(0);
            sampleConfusionMatrix(0);
            sampleClusterAssignments(0);
            sampleLabels(0);
        }
        for (int sampleIndex = 1; sampleIndex < numberOfSamples; sampleIndex++) {
            for (int i = 0; i < numberOfThinningSamples + 1; i++) {
                samplePriors(sampleIndex - 1);
                sampleConfusionMatrix(sampleIndex - 1);
                sampleClusterAssignments(sampleIndex - 1);
                sampleLabels(sampleIndex - 1);
            }
            storeSample(sampleIndex);
        }
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                priorMeans[p] += priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][1]
                            + confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][0];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    labelMeans[p][i] += labelsSamples[sampleNumber][p][i];
            }
        }
        // Compute values for the means and the variances
        for (int p = 0; p < numberOfDomains; p++) {
            priorMeans[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateMeans[p][j] /= numberOfSamples;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelMeans[p][i] /= numberOfSamples;
            for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
                double temp = priorSamples[sampleNumber][p] - priorMeans[p];
                priorVariances[p] += temp * temp;
                for (int j = 0; j < numberOfFunctions; j++) {
                    temp = confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][1]
                            + confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][0]
                            - errorRateMeans[p][j];
                    errorRateVariances[p][j] += temp * temp;
                }
                for (int i = 0; i < numberOfDataSamples[p]; i++) {
                    temp = labelsSamples[sampleNumber][p][i] - labelMeans[p][i];
                    labelVariances[p][i] += temp * temp;
                }
            }
            priorVariances[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateVariances[p][j] /= numberOfSamples;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelVariances[p][i] /= numberOfSamples;
        }
    }

    private void samplePriors(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++)
            priorSamples[sampleNumber][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + priorCounts[p][0], labelsPriorBeta + priorCounts[p][1]);
    }

    private void sampleConfusionMatrix(int sampleNumber) { // TODO: Add checking for number of error rates below chance.
        for (int p = 0; p < numberOfDomains; p++) {
            int currentNumberOfClusters = dpPriors[p].computeClustersDistribution();
            for(int k = 0; k < currentNumberOfClusters; k++) {
                int clusterID = dpPriors[p].getClusterID(k);
                confusionMatrixSamples[sampleNumber][p][clusterID][0][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[0][0] + confusionMatrixCounts[p][clusterID][0][0],
                                                                                                        confusionMatrixPrior[0][1] + confusionMatrixCounts[p][clusterID][0][1]);
                confusionMatrixSamples[sampleNumber][p][clusterID][1][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[1][0] + confusionMatrixCounts[p][clusterID][1][0],
                                                                                                        confusionMatrixPrior[1][1] + confusionMatrixCounts[p][clusterID][1][1]);
                confusionMatrixSamples[sampleNumber][p][clusterID][0][1] = 1 - confusionMatrixSamples[sampleNumber][p][clusterID][0][0];
                confusionMatrixSamples[sampleNumber][p][clusterID][1][1] = 1 - confusionMatrixSamples[sampleNumber][p][clusterID][1][0];
            }
        }
    }

    private void updateCountsBeforeSamplingClusterAssignment(int sampleNumber, int p, int j) {
        for (int i = 0; i < numberOfDataSamples[p]; i++)
            confusionMatrixCounts[p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
        dpPriors[p].removeMemberFromCluster(clusterAssignmentSamples[sampleNumber][p][j]);
    }

    private void updateCountsAfterSamplingClusterAssignment(int sampleNumber, int p, int j) {
        for (int i = 0; i < numberOfDataSamples[p]; i++)
            confusionMatrixCounts[p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
        dpPriors[p].addMemberToCluster(clusterAssignmentSamples[sampleNumber][p][j]);
    }

    private void sampleClusterAssignments(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                updateCountsBeforeSamplingClusterAssignment(sampleNumber, p, j);
                int currentNumberOfClusters = dpPriors[p].computeClustersDistribution();
                double cdf[] = new double[currentNumberOfClusters];
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < currentNumberOfClusters; k++) {
                    int clusterID = dpPriors[p].getClusterID(k);
                    cdf[k] = Math.log(dpPriors[p].getClusterUnnormalizedProbability(clusterID));
                    for (int i = 0; i < numberOfDataSamples[p]; i++)
                        cdf[k] += Math.log(confusionMatrixSamples[sampleNumber][p][clusterID][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]);
                    if (max < cdf[k])
                        max = cdf[k];
                }
                cdf[0] -= max;
                for (int k = 1; k < currentNumberOfClusters; k++) {
                    cdf[k] -= max;
                    cdf[k] = Math.log(Math.exp(cdf[k - 1]) + Math.exp(cdf[k]));
                }
                double uniform = Math.log(random.nextDouble()) + cdf[currentNumberOfClusters - 1];
                int newClusterID = dpPriors[p].getClusterID(currentNumberOfClusters - 1);
                clusterAssignmentSamples[sampleNumber][p][j] = newClusterID;
                for (int k = 0; k < currentNumberOfClusters - 1; k++) {
                    if (cdf[k] > uniform) {
                        int clusterID = dpPriors[p].getClusterID(k);
                        clusterAssignmentSamples[sampleNumber][p][j] = clusterID;
                        break;
                    }
                }
                updateCountsAfterSamplingClusterAssignment(sampleNumber, p, j);
            }
        }
    }

    private void updateCountsBeforeSamplingLabel(int sampleNumber, int p, int i) {
        priorCounts[p][labelsSamples[sampleNumber][p][i]]--;
        for (int j = 0; j < numberOfFunctions; j++)
            confusionMatrixCounts[p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
    }

    private void updateCountsAfterSamplingLabel(int sampleNumber, int p, int i) {
        priorCounts[p][labelsSamples[sampleNumber][p][i]]++;
        for (int j = 0; j < numberOfFunctions; j++)
            confusionMatrixCounts[p][clusterAssignmentSamples[sampleNumber][p][j]][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
    }

    private void sampleLabels(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSamples[sampleNumber][p]; // TODO: Compute this in log-space
                double p1 = priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    p0 *= confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][functionOutputsArray[j][p][i]];
                    p1 *= confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][functionOutputsArray[j][p][i]];
                }
                updateCountsBeforeSamplingLabel(sampleNumber, p, i);
                labelsSamples[sampleNumber][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                updateCountsAfterSamplingLabel(sampleNumber, p, i);
            }
        }
    }

    public double logLikelihood(List<boolean[][]> functionOutputs) {
        return 0; // TODO: Fix this.
    }

    private void storeSample(int sampleIndex) {
        copyArray(priorSamples[sampleIndex - 1], priorSamples[sampleIndex]);
        copyArray(confusionMatrixSamples[sampleIndex - 1], confusionMatrixSamples[sampleIndex]);
        copyArray(clusterAssignmentSamples[sampleIndex - 1], clusterAssignmentSamples[sampleIndex]);
        copyArray(labelsSamples[sampleIndex - 1], labelsSamples[sampleIndex]);
    }

    private void copyArray(Object sourceArray, Object destinationArray) {
        if(sourceArray.getClass().isArray() && destinationArray.getClass().isArray()) {
            for(int i = 0; i < Array.getLength(sourceArray); i++) {
                if(Array.get(sourceArray, i) != null && Array.get(sourceArray, i).getClass().isArray())
                    copyArray(Array.get(sourceArray, i), Array.get(destinationArray, i));
                else
                    Array.set(destinationArray, i, Array.get(sourceArray, i));
            }
        }
    }

    public double[] getPriorMeans() {
        return priorMeans;
    }

    public double[] getPriorVariances() {
        return priorVariances;
    }

    public double[][] getLabelMeans() {
        return labelMeans;
    }

    public double[][] getLabelVariances() {
        return labelVariances;
    }

    public double[][] getErrorRatesMeans() {
        return errorRateMeans;
    }

    public double[][] getErrorRatesVariances() {
        return errorRateVariances;
    }
}
