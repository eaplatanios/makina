package makina.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CoupledBayesianErrorEstimationConfusion {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double labelsPriorAlpha = 1;
    private final double labelsPriorBeta = 1;
    private final double[][] confusionMatrixPrior = new double[][] { new double[] { 10, 1 }, new double[] { 1, 10 } };

    private final int numberOfIterations;
    private final int burnInIterations;
    private final int thinning;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private int numberOfDomains;
    private int[] numberOfDataSamples;
    private int[][][] labelsSamples;
    private int[][][] functionOutputsArray;
    private int[][] clusterAssignmentSamples;
    private final double[][] labelPriorsSamples;
    private final double[][] labelPriorsCounts;       // indexed by domain, 0/1
    private final double[][][][][] confusionMatrixSamples;  // indexed by sample, cluster id, function, 0/1, 0/1
    private final double[][][][] confusionMatrixCounts;     // indexed by cluster id, function, 0/1, 0/1

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;

    private DirichletProcess dpPrior;
    private int maximumNumberOfClusters;

    public CoupledBayesianErrorEstimationConfusion(List<boolean[][]> functionOutputs, int numberOfIterations, int thinning, double alpha) {
        this.numberOfIterations = numberOfIterations;
        burnInIterations = numberOfIterations * 9 / 10;
        this.thinning = thinning;
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
        dpPrior = new DirichletProcess(alpha, numberOfDomains);

        numberOfSamples = (numberOfIterations - burnInIterations) / thinning;
        maximumNumberOfClusters = numberOfDomains;
        labelPriorsSamples = new double[numberOfSamples][numberOfDomains];
        labelPriorsCounts = new double[numberOfDomains][2];
        confusionMatrixSamples = new double[numberOfSamples][maximumNumberOfClusters][numberOfFunctions][2][2];
        confusionMatrixCounts = new double[maximumNumberOfClusters][numberOfFunctions][2][2];
        clusterAssignmentSamples = new int[numberOfSamples][numberOfDomains];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];

        priorMeans = new double[numberOfDomains];
        priorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfDataSamples[p]];
            labelVariances[p] = new double[numberOfDataSamples[p]];
            clusterAssignmentSamples[0][p] = p;
            dpPrior.addMemberToCluster(clusterAssignmentSamples[0][p]);
            labelPriorsSamples[0][p] = 0.5;
            labelsSamples[0][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
                updateCountsAfterSamplingLabel(0, p, i);
            }
        }
        samplePriorsAndBurn(0);
        sampleConfusionMatrixAndBurn(0);
    }

    public void performGibbsSampling() {
        for (int iterationNumber = 0; iterationNumber < burnInIterations; iterationNumber++) {
            samplePriorsAndBurn(0);
            sampleConfusionMatrixAndBurn(0);
            sampleClusterAssignmentsAndBurn(0);
            sampleLabelsAndBurn(0);
        }
        for (int iterationNumber = 0; iterationNumber < numberOfSamples - 1; iterationNumber++) {
            for (int i = 0; i < thinning; i++) {
                samplePriorsAndBurn(iterationNumber);
                sampleConfusionMatrixAndBurn(iterationNumber);
                sampleClusterAssignmentsAndBurn(iterationNumber);
                sampleLabelsAndBurn(iterationNumber);
            }
            samplePriors(iterationNumber);
            sampleConfusionMatrix(iterationNumber);
            sampleClusterAssignments(iterationNumber);
            sampleLabels(iterationNumber);
        }
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                priorMeans[p] += labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    double errorRate = 0;
                    for (int i = 0; i < numberOfDataSamples[p]; i++)
                        errorRate += functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i] ? 1 : 0;
                    errorRateMeans[p][j] += errorRate / numberOfDataSamples[p];
                }
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    errorRateMeans[p][j] += confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][0][1] * labelPriorsSamples[sampleNumber][p]
//                            + confusionMatrixSamples[sampleNumber][p][clusterAssignmentSamples[sampleNumber][p][j]][1][0] * (1 - labelPriorsSamples[sampleNumber][p]);
//                }
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
                double temp = labelPriorsSamples[sampleNumber][p] - priorMeans[p];
                priorVariances[p] += temp * temp;
                for (int j = 0; j < numberOfFunctions; j++) {
                    temp = (confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j][0][1] * labelPriorsSamples[sampleNumber][p]
                            + confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j][1][0] * (1 - labelPriorsSamples[sampleNumber][p]))
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

    private void updateCountsBeforeSamplingLabel(int sampleNumber, int p, int i) {
        labelPriorsCounts[p][labelsSamples[sampleNumber][p][i]]--;
        for (int j = 0; j < numberOfFunctions; j++)
            confusionMatrixCounts[clusterAssignmentSamples[sampleNumber][p]][j][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
    }

    private void updateCountsAfterSamplingLabel(int sampleNumber, int p, int i) {
        labelPriorsCounts[p][labelsSamples[sampleNumber][p][i]]++;
        for (int j = 0; j < numberOfFunctions; j++)
            confusionMatrixCounts[clusterAssignmentSamples[sampleNumber][p]][j][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
    }

    private void samplePriorsAndBurn(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++)
            labelPriorsSamples[sampleNumber][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelPriorsCounts[p][1], labelsPriorBeta + labelPriorsCounts[p][0]);
    }

    private void samplePriors(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelPriorsSamples[iterationNumber + 1][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelPriorsCounts[p][1], labelsPriorBeta + labelPriorsCounts[p][0]);
        }
    }

    private void sampleConfusionMatrixAndBurn(int sampleNumber) {
        for (int k = 0; k< maximumNumberOfClusters; k++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                confusionMatrixSamples[sampleNumber][k][j][0][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[0][0] + confusionMatrixCounts[k][j][0][0],
                                                                                                confusionMatrixPrior[0][1] + confusionMatrixCounts[k][j][0][1]);
                confusionMatrixSamples[sampleNumber][k][j][1][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[1][0] + confusionMatrixCounts[k][j][1][0],
                                                                                                confusionMatrixPrior[1][1] + confusionMatrixCounts[k][j][1][1]);
                confusionMatrixSamples[sampleNumber][k][j][0][1] = 1 - confusionMatrixSamples[sampleNumber][k][j][0][0];
                confusionMatrixSamples[sampleNumber][k][j][1][1] = 1 - confusionMatrixSamples[sampleNumber][k][j][1][0];
            }
        }
    }

    private void sampleConfusionMatrix(int sampleNumber) {
        for (int k = 0; k< maximumNumberOfClusters; k++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                confusionMatrixSamples[sampleNumber + 1][k][j][0][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[0][0] + confusionMatrixCounts[k][j][0][0],
                                                                                                    confusionMatrixPrior[0][1] + confusionMatrixCounts[k][j][0][1]);
                confusionMatrixSamples[sampleNumber + 1][k][j][1][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[1][0] + confusionMatrixCounts[k][j][1][0],
                                                                                                    confusionMatrixPrior[1][1] + confusionMatrixCounts[k][j][1][1]);
                confusionMatrixSamples[sampleNumber + 1][k][j][0][1] = 1 - confusionMatrixSamples[sampleNumber][k][j][0][0];
                confusionMatrixSamples[sampleNumber + 1][k][j][1][1] = 1 - confusionMatrixSamples[sampleNumber][k][j][1][0];
            }
        }
    }

    private void updateCountsBeforeSamplingClusterAssignment(int sampleNumber, int p) {
        for (int j = 0; j < numberOfFunctions; j++)
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                confusionMatrixCounts[clusterAssignmentSamples[sampleNumber][p]][j][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
        dpPrior.removeMemberFromCluster(clusterAssignmentSamples[sampleNumber][p]);
    }

    private void updateCountsAfterSamplingClusterAssignment(int sampleNumber, int p) {
        for (int j = 0; j < numberOfFunctions; j++)
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                confusionMatrixCounts[clusterAssignmentSamples[sampleNumber][p]][j][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
        dpPrior.addMemberToCluster(clusterAssignmentSamples[sampleNumber][p]);
    }

    private void sampleClusterAssignmentsAndBurn(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dpPrior.removeMemberFromCluster(clusterAssignmentSamples[sampleNumber][p]);
            int currentNumberOfClusters = dpPrior.computeClustersDistribution();
            double cdf[] = new double[currentNumberOfClusters];
            for(int i = 0; i < currentNumberOfClusters; i++)
                cdf[i] = Math.log(dpPrior.getClusterUnnormalizedProbability(i));
            updateCountsBeforeSamplingClusterAssignment(sampleNumber, p);
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < numberOfFunctions; j++) {
                int cnt_errs[][] = new int[2][2];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    cnt_errs[labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
                for (int k = 0; k < currentNumberOfClusters; k++) {
                    int clusterID = dpPrior.getClusterID(k);
                    cdf[k] = Math.log(dpPrior.getClusterUnnormalizedProbability(k));
                    cdf[k] += cnt_errs[0][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][j][0][0]);
                    cdf[k] += cnt_errs[0][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][j][0][1]);
                    cdf[k] += cnt_errs[1][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][j][1][0]);
                    cdf[k] += cnt_errs[1][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][j][1][1]);
                    if (max < cdf[k])
                        max = cdf[k];
                }
            }
            cdf[0] -= max;
            for (int k = 1; k < currentNumberOfClusters; k++) {
                cdf[k] -= max;
                cdf[k] = Math.log(Math.exp(cdf[k - 1]) + Math.exp(cdf[k]));
            }
            double uniform = Math.log(random.nextDouble()) + cdf[currentNumberOfClusters - 1];
            int newClusterID = dpPrior.getClusterID(currentNumberOfClusters - 1);
            clusterAssignmentSamples[sampleNumber][p] = newClusterID;
            for (int k = 0; k < currentNumberOfClusters - 1; k++) {
                if (cdf[k] > uniform) {
                    int clusterID = dpPrior.getClusterID(k);
                    clusterAssignmentSamples[sampleNumber][p] = clusterID;
                    break;
                }
            }
            updateCountsAfterSamplingClusterAssignment(sampleNumber, p);
        }
    }

    private void sampleClusterAssignments(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dpPrior.removeMemberFromCluster(clusterAssignmentSamples[sampleNumber][p]);
            int currentNumberOfClusters = dpPrior.computeClustersDistribution();
            double cdf[] = new double[currentNumberOfClusters];
            for (int i = 0; i < currentNumberOfClusters; i++)
                cdf[i] = Math.log(dpPrior.getClusterUnnormalizedProbability(i));
            updateCountsBeforeSamplingClusterAssignment(sampleNumber, p);
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < numberOfFunctions; j++) {
                int cnt_errs[][] = new int[2][2];
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    cnt_errs[labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
                for (int k = 0; k < currentNumberOfClusters; k++) {
                    int clusterID = dpPrior.getClusterID(k);
                    cdf[k] = Math.log(dpPrior.getClusterUnnormalizedProbability(k));
                    cdf[k] += cnt_errs[0][0] * Math.log(confusionMatrixSamples[sampleNumber + 1][clusterID][j][0][0]);
                    cdf[k] += cnt_errs[0][1] * Math.log(confusionMatrixSamples[sampleNumber + 1][clusterID][j][0][1]);
                    cdf[k] += cnt_errs[1][0] * Math.log(confusionMatrixSamples[sampleNumber + 1][clusterID][j][1][0]);
                    cdf[k] += cnt_errs[1][1] * Math.log(confusionMatrixSamples[sampleNumber + 1][clusterID][j][1][1]);
                    if (max < cdf[k])
                        max = cdf[k];
                }
            }
            cdf[0] -= max;
            for (int k = 1; k < currentNumberOfClusters; k++) {
                cdf[k] -= max;
                cdf[k] = Math.log(Math.exp(cdf[k - 1]) + Math.exp(cdf[k]));
            }
            double uniform = Math.log(random.nextDouble()) + cdf[currentNumberOfClusters - 1];
            int newClusterID = dpPrior.getClusterID(currentNumberOfClusters - 1);
            clusterAssignmentSamples[sampleNumber + 1][p] = newClusterID;
            for (int k = 0; k < currentNumberOfClusters - 1; k++) {
                if (cdf[k] > uniform) {
                    int clusterID = dpPrior.getClusterID(k);
                    clusterAssignmentSamples[sampleNumber + 1][p] = clusterID;
                    break;
                }
            }
            updateCountsAfterSamplingClusterAssignment(sampleNumber, p);
        }
    }

    private void sampleLabelsAndBurn(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - labelPriorsSamples[sampleNumber][p]; // TODO: Compute this in log-space
                double p1 = labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    p0 *= confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j][0][functionOutputsArray[j][p][i]];
                    p1 *= confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j][1][functionOutputsArray[j][p][i]];
                }
                int newLabel = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                if (labelsSamples[sampleNumber][p][i] != newLabel) {
                    updateCountsBeforeSamplingLabel(sampleNumber, p, i);
                    labelsSamples[sampleNumber][p][i] = newLabel;
                    updateCountsAfterSamplingLabel(sampleNumber, p, i);
                }
            }
        }
    }

    private void sampleLabels(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[sampleNumber + 1][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - labelPriorsSamples[sampleNumber + 1][p]; // TODO: Compute this in log-space
                double p1 = labelPriorsSamples[sampleNumber + 1][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    p0 *= confusionMatrixSamples[sampleNumber + 1][clusterAssignmentSamples[sampleNumber + 1][p]][j][0][functionOutputsArray[j][p][i]];
                    p1 *= confusionMatrixSamples[sampleNumber + 1][clusterAssignmentSamples[sampleNumber + 1][p]][j][1][functionOutputsArray[j][p][i]];
                }
                int newLabel = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                if (labelsSamples[sampleNumber][p][i] != newLabel) {
                    updateCountsBeforeSamplingLabel(sampleNumber, p, i);
                    labelsSamples[sampleNumber + 1][p][i] = newLabel;
                    updateCountsAfterSamplingLabel(sampleNumber + 1, p, i);
                } else {
                    labelsSamples[sampleNumber + 1][p][i] = labelsSamples[sampleNumber][p][i];
                }
            }
        }
    }

    public double logLikelihood(List<boolean[][]> functionOutputs) {
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
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[0][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
            }
        }
        double logLikelihood = 0;
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            sampleLabelsAndBurn(sampleNumber);
            Map<Integer, AtomicInteger> clusterCounts = new HashMap<>();
            for (int p = 0; p < numberOfDomains; p++) {
                if (!clusterCounts.containsKey(clusterAssignmentSamples[sampleNumber][p]))
                    clusterCounts.put(clusterAssignmentSamples[sampleNumber][p], new AtomicInteger(1));
                else
                    clusterCounts.get(clusterAssignmentSamples[sampleNumber][p]).incrementAndGet();
            }
            for (int p = 0; p < numberOfDomains; p++) {
                // Label prior term
                logLikelihood += (labelsPriorAlpha - 1) * Math.log(labelPriorsSamples[sampleNumber][p])
                        + (labelsPriorBeta - 1) * Math.log(1 - labelPriorsSamples[sampleNumber][p]);
                // Cluster assignments term
                logLikelihood += Math.log(clusterCounts.get(clusterAssignmentSamples[sampleNumber][p]).intValue()) - Math.log(numberOfDomains);
                // Labels term
                for (int i = 0; i < numberOfDataSamples[p]; i++) {
                    if (labelsSamples[sampleNumber][p][i] == 1)
                        logLikelihood += Math.log(labelPriorsSamples[sampleNumber][p]);
                    else
                        logLikelihood += Math.log(1 - labelPriorsSamples[sampleNumber][p]);
                }
                // Confusion matrix term
                for (int clusterID : clusterCounts.keySet()) {
                    for (int j = 0; j < numberOfFunctions; j++) {
                        logLikelihood += confusionMatrixPrior[0][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][j][0][0]);
                        logLikelihood += confusionMatrixPrior[0][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][j][0][1]);
                        logLikelihood += confusionMatrixPrior[1][0] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][j][1][0]);
                        logLikelihood += confusionMatrixPrior[1][1] * Math.log(confusionMatrixSamples[sampleNumber][clusterID][j][1][1]);
                    }
                }
                // Function outputs term
                for (int j = 0; j < numberOfFunctions; j++)
                    for (int i = 0; i < numberOfDataSamples[p]; i++)
                        logLikelihood += Math.log(confusionMatrixSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]);
            }
        }
        return logLikelihood / numberOfSamples;
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
