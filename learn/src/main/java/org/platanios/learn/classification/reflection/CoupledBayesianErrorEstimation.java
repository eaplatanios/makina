package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.platanios.learn.math.matrix.MatrixUtilities;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import static org.apache.commons.math3.special.Beta.logBeta;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CoupledBayesianErrorEstimation {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double labelsPriorAlpha = 1;
    private final double labelsPriorBeta = 1;
    private final double errorRatesPriorAlpha = 1;
    private final double errorRatesPriorBeta = 10;

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
    private final double[][] priorSamples;
    private final double[][][] errorRateSamples;

    private double[][] disagreements;
    private double[][] sum_1;
    private double[][] sum_2;

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;
    
    private DirichletProcess dpPrior;

    public CoupledBayesianErrorEstimation(List<boolean[][]> functionOutputs, int numberOfIterations, int thinning, double alpha) {
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
        priorSamples = new double[numberOfSamples][numberOfDomains];
        errorRateSamples = new double[numberOfSamples][numberOfDomains][numberOfFunctions];
        clusterAssignmentSamples = new int[numberOfSamples][numberOfDomains];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];

        disagreements = new double[numberOfFunctions][numberOfDomains];
        sum_1 = new double[numberOfFunctions][numberOfDomains];
        sum_2 = new double[numberOfFunctions][numberOfDomains];

        priorMeans = new double[numberOfDomains];
        priorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfDataSamples[p]];
            labelVariances[p] = new double[numberOfDataSamples[p]];
            clusterAssignmentSamples[0][p] = 0;
            dpPrior.addMemberToCluster(0);
            priorSamples[0][p] = 0.5;
            labelsSamples[0][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
//                labelsSamples[0][p][i] = randomDataGenerator.nextBinomial(1, 0.5);
            }
            for (int j = 0; j < numberOfFunctions; j++) {
                errorRateSamples[0][p][j] = 0.25;
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[0][p][i])
                        disagreements[j][p]++;
            }
        }
        for (int k = 0; k < numberOfDomains; k++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                sum_1[j][k] = 0;
                sum_2[j][k] = 0;
                for (int p = 0; p < numberOfDomains; p++) {
                    if (clusterAssignmentSamples[0][p] == k) {
                        sum_1[j][k] += numberOfDataSamples[p];
                        sum_2[j][k] += disagreements[j][p];
                    }
                }
            }
        }
    }

    public void performGibbsSampling() {
        for (int iterationNumber = 0; iterationNumber < 100; iterationNumber++)
            sampleZAndBurnWithCollapsedErrorRates(0);
        for (int iterationNumber = 0; iterationNumber < burnInIterations; iterationNumber++) {
            if (iterationNumber < burnInIterations / 2) {
                samplePriorsAndBurn(0);
                sampleLabelsAndBurnWithCollapsedErrorRates(0);
                sampleZAndBurnWithCollapsedErrorRates(0);
            } else {
                samplePriorsAndBurn(0);
                sampleErrorRatesAndBurn(0);
                sampleZAndBurn(0);
                sampleLabelsAndBurn(0);
            }
        }
        for (int iterationNumber = 0; iterationNumber < numberOfSamples - 1; iterationNumber++) {
            for (int i = 0; i < thinning; i++) {
                samplePriorsAndBurn(iterationNumber);
                sampleErrorRatesAndBurn(iterationNumber);
                sampleZAndBurn(iterationNumber);
                sampleLabelsAndBurn(iterationNumber);
            }
            samplePriors(iterationNumber, true);
            sampleErrorRates(iterationNumber, true);
            sampleZ(iterationNumber);
            sampleLabels(iterationNumber);
        }
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                priorMeans[p] += priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRateSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j];
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
                    temp = errorRateSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j] - errorRateMeans[p][j];
                    errorRateVariances[p][j] += temp * temp;
                }
                for (int i = 0; i < numberOfDataSamples[p]; i++) {
                    temp = labelsSamples[sampleNumber][p][i] - labelMeans[p][i];
                    labelVariances[p][i] += temp * temp;
                }
            }
            priorVariances[p] /= (numberOfIterations - burnInIterations - 1);
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateVariances[p][j] /= (numberOfIterations - burnInIterations - 1);
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelVariances[p][i] /= (numberOfIterations - burnInIterations - 1);
        }
    }

    private void samplePriorsAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[iterationNumber][p][i];
            priorSamples[iterationNumber][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelsCount, labelsPriorBeta + numberOfDataSamples[p] - labelsCount);
        }
    }

    private void samplePriors(int iterationNumber) {
        samplePriors(iterationNumber, false);
    }

    private void samplePriors(int iterationNumber, boolean sampleMean) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[iterationNumber][p][i];
            if (sampleMean)
                priorSamples[iterationNumber + 1][p] = (labelsPriorAlpha + labelsCount) / (labelsPriorAlpha + labelsPriorBeta + numberOfDataSamples[p]);
            else
                priorSamples[iterationNumber + 1][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelsCount, labelsPriorBeta + numberOfDataSamples[p] - labelsCount);
        }
    }

    private void sampleErrorRatesAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                int zCount = 0;
                for (int k = 0; k < numberOfDomains; k++) {
                    if (clusterAssignmentSamples[iterationNumber][k] == clusterAssignmentSamples[iterationNumber][p]) {
                        for (int i = 0; i < numberOfDataSamples[k]; i++)
                            if (functionOutputsArray[j][k][i] != labelsSamples[iterationNumber][k][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[k];
                    }
                }
                errorRateSamples[iterationNumber][p][j] = randomDataGenerator.nextBeta(errorRatesPriorAlpha + disagreementCount, errorRatesPriorBeta + zCount - disagreementCount);
                if (errorRateSamples[iterationNumber][p][j] < 0.5)
                    numberOfErrorRatesBelowChance += 1;
            }
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0)
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateSamples[iterationNumber][p][j] = 1 - errorRateSamples[iterationNumber][p][j];
        }
    }

    private void sampleErrorRates(int iterationNumber) {
        sampleErrorRates(iterationNumber, false);
    }

    private void sampleErrorRates(int iterationNumber, boolean sampleMean) {
        for (int p = 0; p < numberOfDomains; p++) {
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                int zCount = 0;
                for (int k = 0; k < numberOfDomains; k++) {
                    if (clusterAssignmentSamples[iterationNumber][k] == clusterAssignmentSamples[iterationNumber][p]) {
                        for (int i = 0; i < numberOfDataSamples[k]; i++)
                            if (functionOutputsArray[j][k][i] != labelsSamples[iterationNumber][k][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[k];
                    }
                }
                if (sampleMean)
                    errorRateSamples[iterationNumber + 1][p][j] = (errorRatesPriorAlpha + disagreementCount) / (errorRatesPriorAlpha + errorRatesPriorBeta + zCount);
                else
                    errorRateSamples[iterationNumber + 1][p][j] = randomDataGenerator.nextBeta(errorRatesPriorAlpha + disagreementCount, errorRatesPriorBeta + zCount - disagreementCount);
                if (errorRateSamples[iterationNumber + 1][p][j] < 0.5)
                    numberOfErrorRatesBelowChance += 1;
            }
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0) {
                priorSamples[iterationNumber + 1][p] = 1 - priorSamples[iterationNumber + 1][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateSamples[iterationNumber + 1][p][j] = 1 - errorRateSamples[iterationNumber + 1][p][j];
            }
        }
    }

    private void sampleZAndBurnWithCollapsedErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dpPrior.removeMemberFromCluster(clusterAssignmentSamples[iterationNumber][p]);
            int currentNumberOfClusters = dpPrior.computeClustersDistribution();
            double z_probabilities[] = new double[currentNumberOfClusters];
            for(int i = 0; i < currentNumberOfClusters; i++)
                z_probabilities[i] = Math.log(dpPrior.getClusterUnnormalizedProbability(i));
            for (int j = 0; j < numberOfFunctions; j++) {
                sum_1[j][clusterAssignmentSamples[iterationNumber][p]] -= numberOfDataSamples[p];
                sum_2[j][clusterAssignmentSamples[iterationNumber][p]] -= disagreements[j][p];
                for(int i = 0; i < currentNumberOfClusters - 1; i++) {
                    int clusterID = dpPrior.getClusterID(i);
                    z_probabilities[i] +=
                            logBeta(errorRatesPriorAlpha + sum_2[j][clusterID] + disagreements[j][p],
                                    errorRatesPriorBeta + sum_1[j][clusterID] - sum_2[j][clusterID] + numberOfDataSamples[p] - disagreements[j][p])
                                    - logBeta(errorRatesPriorAlpha + sum_2[j][clusterID], errorRatesPriorBeta + sum_1[j][clusterID] - sum_2[j][clusterID]);
                }
                z_probabilities[currentNumberOfClusters - 1] +=
                        logBeta(errorRatesPriorAlpha + disagreements[j][p], errorRatesPriorBeta + numberOfDataSamples[p] - disagreements[j][p])
                                - logBeta(errorRatesPriorAlpha, errorRatesPriorBeta);
            }
            for (int i = 1; i < currentNumberOfClusters; i++)
                z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i - 1], z_probabilities[i]);
            double uniform = Math.log(random.nextDouble()) + z_probabilities[currentNumberOfClusters - 1];
            int newClusterID = dpPrior.getClusterID(currentNumberOfClusters - 1);
            clusterAssignmentSamples[iterationNumber][p] = newClusterID;
            for(int i = 0; i < currentNumberOfClusters - 1; i++)  {
                if (z_probabilities[i] > uniform) {
                    int clusterID = dpPrior.getClusterID(i);
                    clusterAssignmentSamples[iterationNumber][p] = clusterID;
                    for (int j = 0; j < numberOfFunctions; j++) {
                        sum_1[j][clusterID] += numberOfDataSamples[p];
                        sum_2[j][clusterID] += disagreements[j][p];
                    }
                    dpPrior.addMemberToCluster(clusterID);
                    break;
                }
            }
            if (clusterAssignmentSamples[iterationNumber][p] == newClusterID) {
                for (int j = 0; j < numberOfFunctions; j++) {
                    sum_1[j][newClusterID] += numberOfDataSamples[p];
                    sum_2[j][newClusterID] += disagreements[j][p];
                }
                dpPrior.addMemberToCluster(newClusterID);
            }
        }
    }

    private void sampleZWithCollapsedErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dpPrior.removeMemberFromCluster(clusterAssignmentSamples[iterationNumber][p]);
            int currentNumberOfClusters = dpPrior.computeClustersDistribution();
            double z_probabilities[] = new double[currentNumberOfClusters];
            for(int i = 0; i < currentNumberOfClusters; i++)
                z_probabilities[i] = Math.log(dpPrior.getClusterUnnormalizedProbability(i));
            for (int j = 0; j < numberOfFunctions; j++) {
                sum_1[j][clusterAssignmentSamples[iterationNumber][p]] -= numberOfDataSamples[p];
                sum_2[j][clusterAssignmentSamples[iterationNumber][p]] -= disagreements[j][p];
                for(int i = 0; i < currentNumberOfClusters - 1; i++) {
                    int clusterID = dpPrior.getClusterID(i);
                    z_probabilities[i] +=
                            logBeta(errorRatesPriorAlpha + sum_2[j][clusterID] + disagreements[j][p],
                                    errorRatesPriorBeta + sum_1[j][clusterID] - sum_2[j][clusterID] + numberOfDataSamples[p] - disagreements[j][p])
                                    - logBeta(errorRatesPriorAlpha + sum_2[j][clusterID], errorRatesPriorBeta + sum_1[j][clusterID] - sum_2[j][clusterID]);
                }
                z_probabilities[currentNumberOfClusters - 1] +=
                        logBeta(errorRatesPriorAlpha + disagreements[j][p], errorRatesPriorBeta + numberOfDataSamples[p] - disagreements[j][p])
                                - logBeta(errorRatesPriorAlpha, errorRatesPriorBeta);
            }
            for (int i = 1; i < currentNumberOfClusters; i++)
                z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i - 1], z_probabilities[i]);
            double uniform = Math.log(random.nextDouble()) + z_probabilities[currentNumberOfClusters - 1];
            int newClusterID = dpPrior.getClusterID(currentNumberOfClusters - 1);
            clusterAssignmentSamples[iterationNumber + 1][p] = newClusterID;
            for(int i = 0; i < currentNumberOfClusters - 1; i++)  {
                if (z_probabilities[i] > uniform) {
                    int clusterID = dpPrior.getClusterID(i);
                    clusterAssignmentSamples[iterationNumber + 1][p] = clusterID;
                    for (int j = 0; j < numberOfFunctions; j++) {
                        sum_1[j][clusterID] += numberOfDataSamples[p];
                        sum_2[j][clusterID] += disagreements[j][p];
                    }
                    dpPrior.addMemberToCluster(clusterID);
                    break;
                }
            }
            if (clusterAssignmentSamples[iterationNumber + 1][p] == newClusterID) {
                for (int j = 0; j < numberOfFunctions; j++) {
                    sum_1[j][newClusterID] += numberOfDataSamples[p];
                    sum_2[j][newClusterID] += disagreements[j][p];
                }
                dpPrior.addMemberToCluster(newClusterID);
            }
        }
    }

    private void sampleZAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dpPrior.removeMemberFromCluster(clusterAssignmentSamples[iterationNumber][p]);
            int currentNumberOfClusters = dpPrior.computeClustersDistribution();
            double z_probabilities[] = new double[currentNumberOfClusters];
            for(int i = 0; i < currentNumberOfClusters; i++)
                z_probabilities[i] = Math.log(dpPrior.getClusterUnnormalizedProbability(i));
            for (int j = 0; j < numberOfFunctions; j++) {
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
                for(int i = 0; i < currentNumberOfClusters - 1; i++)  {
                    int clusterID = dpPrior.getClusterID(i);
                    z_probabilities[i] += disagreements[j][p] * Math.log(errorRateSamples[iterationNumber][clusterID][j]);
                    z_probabilities[i] += (numberOfDataSamples[p] - disagreements[j][p])
                            * Math.log(1 - errorRateSamples[iterationNumber][clusterID][j]);
                }
                z_probabilities[currentNumberOfClusters - 1] +=
                        logBeta(errorRatesPriorAlpha + disagreements[j][p],
                                errorRatesPriorBeta + numberOfDataSamples[p] - disagreements[j][p])
                                - logBeta(errorRatesPriorAlpha, errorRatesPriorBeta);
            }
            for (int i = 1; i < currentNumberOfClusters; i++)
                z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i - 1], z_probabilities[i]);
            double uniform = Math.log(random.nextDouble()) + z_probabilities[currentNumberOfClusters - 1];
            int newClusterID = dpPrior.getClusterID(currentNumberOfClusters - 1);
            clusterAssignmentSamples[iterationNumber][p] = newClusterID;
            for(int i = 0; i < currentNumberOfClusters - 1; i++)  {
                if(z_probabilities[i] > uniform) {
                    int clusterID = dpPrior.getClusterID(i);
                    clusterAssignmentSamples[iterationNumber][p] = clusterID;
                    dpPrior.addMemberToCluster(clusterID);
                    break;
                }
            }
            if (clusterAssignmentSamples[iterationNumber][p] == newClusterID) {
                dpPrior.addMemberToCluster(newClusterID);
//                int numberOfErrorRatesBelowChance = 0;
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    errorRateSamples[iterationNumber][dpPrior.clustersDistribution[totalMembersCount - 1].topic][j] = randomDataGenerator.nextBeta(errorRatesPriorAlpha + disagreements[j][p], errorRatesPriorBeta + numberOfDataSamples[p] - disagreements[j][p]);
//                    if (errorRateSamples[iterationNumber][dpPrior.clustersDistribution[totalMembersCount - 1].topic][j] < 0.5)
//                        numberOfErrorRatesBelowChance += 1;
//                }
//                if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0)
//                    for (int j = 0; j < numberOfFunctions; j++)
//                        errorRateSamples[iterationNumber][dpPrior.clustersDistribution[totalMembersCount - 1].topic][j] = 1 - errorRateSamples[iterationNumber][dpPrior.clustersDistribution[totalMembersCount - 1].topic][j];
            }
        }
    }

    private void sampleZ(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            dpPrior.removeMemberFromCluster(clusterAssignmentSamples[iterationNumber][p]);
            int currentNumberOfClusters = dpPrior.computeClustersDistribution();
            double z_probabilities[] = new double[currentNumberOfClusters];
            for(int i = 0; i < currentNumberOfClusters; i++)
                z_probabilities[i] = Math.log(dpPrior.getClusterUnnormalizedProbability(i));
            for (int j = 0; j < numberOfFunctions; j++) {
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
                for(int i=0;i<currentNumberOfClusters - 1;i++) {
                    int clusterID = dpPrior.getClusterID(i);
                    z_probabilities[i] += disagreements[j][p] * Math.log(errorRateSamples[iterationNumber + 1][clusterID][j]);
                    z_probabilities[i] += (numberOfDataSamples[p] - disagreements[j][p])
                            * Math.log(1 - errorRateSamples[iterationNumber + 1][clusterID][j]);
                }
                z_probabilities[currentNumberOfClusters - 1] += logBeta(errorRatesPriorAlpha + disagreements[j][p],
                                                                        errorRatesPriorBeta + numberOfDataSamples[p] - disagreements[j][p])
                        - logBeta(errorRatesPriorAlpha, errorRatesPriorBeta);
            }
            for(int i=1;i<currentNumberOfClusters;i++){
                z_probabilities[i] = MatrixUtilities.computeLogSumExp(z_probabilities[i-1],z_probabilities[i]);
            }

            double uniform = Math.log(random.nextDouble()) + z_probabilities[currentNumberOfClusters - 1];
            int newClusterID = dpPrior.getClusterID(currentNumberOfClusters - 1);
            clusterAssignmentSamples[iterationNumber + 1][p] = newClusterID;
            for (int i = 0; i < currentNumberOfClusters - 1; i++) {
                if (z_probabilities[i] > uniform) {
                    int clusterID = dpPrior.getClusterID(i);
                    clusterAssignmentSamples[iterationNumber + 1][p] = clusterID;
                    dpPrior.addMemberToCluster(clusterID);
                    break;
                }
            }
            if (clusterAssignmentSamples[iterationNumber + 1][p] == newClusterID) {
                dpPrior.addMemberToCluster(newClusterID);
//                int numberOfErrorRatesBelowChance = 0;
//                for (int j = 0; j < numberOfFunctions; j++) {
//                    errorRateSamples[iterationNumber + 1][dpPrior.clustersDistribution[totalMembersCount - 1].topic][j] = randomDataGenerator.nextBeta(errorRatesPriorAlpha + disagreements[j][p], errorRatesPriorBeta + numberOfDataSamples[p] - disagreements[j][p]);
//                    if (errorRateSamples[iterationNumber + 1][dpPrior.clustersDistribution[totalMembersCount - 1].topic][j] < 0.5)
//                        numberOfErrorRatesBelowChance += 1;
//                }
//                if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0)
//                    for (int j = 0; j < numberOfFunctions; j++)
//                        errorRateSamples[iterationNumber + 1][dpPrior.clustersDistribution[totalMembersCount - 1].topic][j] = 1 - errorRateSamples[iterationNumber + 1][dpPrior.clustersDistribution[totalMembersCount - 1].topic][j];
            }
        }
    }

    private void sampleLabelsAndBurnWithCollapsedErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++)
                sum_2[j][clusterAssignmentSamples[iterationNumber][p]] -= disagreements[j][p];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double a1 = errorRatesPriorAlpha;
                double a0 = errorRatesPriorBeta;
                double sum1 = 0;
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i]) {
                        a1 += --disagreements[j][p] + sum_2[j][clusterAssignmentSamples[iterationNumber][p]];
                        a0 += sum_1[j][clusterAssignmentSamples[iterationNumber][p]] - sum_2[j][clusterAssignmentSamples[iterationNumber][p]] - disagreements[j][p];
                    }
                    if (functionOutputsArray[j][p][i] != 1)
                        sum1++;
                }
                double p1 = priorSamples[iterationNumber][p];
                double p0 = 1 - priorSamples[iterationNumber][p];
                for (int m = 0; m < sum1; m++) {
                    p1 *= a1 + m;
                    p0 *= a0 + m;
                }
                for (int m = 0; m < numberOfFunctions - sum1; m++) {
                    p1 *= a0 + m;
                    p0 *= a1 + m;
                }
                labelsSamples[iterationNumber][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                for (int j = 0; j < numberOfFunctions; j++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
            }
            for (int j = 0; j < numberOfFunctions; j++)
                sum_2[j][clusterAssignmentSamples[iterationNumber][p]] += disagreements[j][p];
        }
    }

    private void sampleLabelsWithCollapsedErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[iterationNumber + 1][p] = new int[numberOfDataSamples[p]];
            for (int j = 0; j < numberOfFunctions; j++)
                sum_2[j][clusterAssignmentSamples[iterationNumber + 1][p]] -= disagreements[j][p];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double a1 = errorRatesPriorAlpha;
                double a0 = errorRatesPriorBeta;
                double sum1 = 0;
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i]) {
                        a1 += --disagreements[j][p] + sum_2[j][clusterAssignmentSamples[iterationNumber + 1][p]];
                        a0 += sum_1[j][clusterAssignmentSamples[iterationNumber + 1][p]] - sum_2[j][clusterAssignmentSamples[iterationNumber + 1][p]] - disagreements[j][p];
                    }
                    if (functionOutputsArray[j][p][i] != 1)
                        sum1++;
                }
                double p1 = priorSamples[iterationNumber + 1][p];
                double p0 = 1 - priorSamples[iterationNumber + 1][p];
                for (int m = 0; m < sum1; m++) {
                    p1 *= a1 + m;
                    p0 *= a0 + m;
                }
                for (int m = 0; m < numberOfFunctions - sum1; m++) {
                    p1 *= a0 + m;
                    p0 *= a1 + m;
                }
                labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                for (int j = 0; j < numberOfFunctions; j++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber + 1][p][i])
                        disagreements[j][p]++;
            }
            for (int j = 0; j < numberOfFunctions; j++)
                sum_2[j][clusterAssignmentSamples[iterationNumber + 1][p]] += disagreements[j][p];
        }
    }

    private void sampleLabelsAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[iterationNumber][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSamples[iterationNumber][p]; // TODO: Compute this in log-space
                double p1 = priorSamples[iterationNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRateSamples[iterationNumber][clusterAssignmentSamples[iterationNumber][p]][j]);
                        p1 *= errorRateSamples[iterationNumber][clusterAssignmentSamples[iterationNumber][p]][j];
                    } else {
                        p0 *= errorRateSamples[iterationNumber][clusterAssignmentSamples[iterationNumber][p]][j];
                        p1 *= (1 - errorRateSamples[iterationNumber][clusterAssignmentSamples[iterationNumber][p]][j]);
                    }
                }
                labelsSamples[iterationNumber][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
            }
        }
    }

    private void sampleLabels(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[iterationNumber + 1][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSamples[iterationNumber + 1][p]; // TODO: Compute this in log-space
                double p1 = priorSamples[iterationNumber + 1][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRateSamples[iterationNumber + 1][clusterAssignmentSamples[iterationNumber + 1][p]][j]);
                        p1 *= errorRateSamples[iterationNumber + 1][clusterAssignmentSamples[iterationNumber + 1][p]][j];
                    } else {
                        p0 *= errorRateSamples[iterationNumber + 1][clusterAssignmentSamples[iterationNumber + 1][p]][j];
                        p1 *= (1 - errorRateSamples[iterationNumber + 1][clusterAssignmentSamples[iterationNumber + 1][p]][j]);
                    }
                }
                labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
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
                logLikelihood += (labelsPriorAlpha - 1) * Math.log(priorSamples[sampleNumber][p])
                        + (labelsPriorBeta - 1) * Math.log(1 - priorSamples[sampleNumber][p]);
                // Cluster assignments term
                logLikelihood += Math.log(clusterCounts.get(clusterAssignmentSamples[sampleNumber][p]).intValue()) - Math.log(numberOfDomains);
                // Labels term
                for (int i = 0; i < numberOfDataSamples[p]; i++) {
                    if (labelsSamples[sampleNumber][p][i] == 1)
                        logLikelihood += Math.log(priorSamples[sampleNumber][p]);
                    else
                        logLikelihood += Math.log(1 - priorSamples[sampleNumber][p]);
                }
                // Error rates term
                for (int j = 0; j < numberOfFunctions; j++)
                    logLikelihood += (errorRatesPriorAlpha - 1) * Math.log(errorRateSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j])
                            + (errorRatesPriorBeta - 1) * Math.log(1 - errorRateSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j]);
                // Function outputs term
                for (int j = 0; j < numberOfFunctions; j++)
                    for (int i = 0; i < numberOfDataSamples[p]; i++) {
                        if (functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i])
                            logLikelihood += Math.log(errorRateSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j]);
                        else
                            logLikelihood += Math.log(1 - errorRateSamples[sampleNumber][clusterAssignmentSamples[sampleNumber][p]][j]);
                    }
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
