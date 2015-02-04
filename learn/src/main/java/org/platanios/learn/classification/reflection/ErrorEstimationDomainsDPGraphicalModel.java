package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.platanios.learn.math.matrix.MatrixUtilities;

import java.util.List;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationDomainsDPGraphicalModel {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double alpha = 1000000;
    private final double alpha_p = 1;
    private final double beta_p = 1;
    private final double alpha_e = 1;
    private final double beta_e = 1;

    private final int numberOfIterations;
    private final int burnInIterations;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfSamples;
    private final int[][][] labelsSamples;
    private final int[][][] functionOutputsArray;
    private final int[][] zSamples;
    private final double[][][] phiSamples;

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;

    public ErrorEstimationDomainsDPGraphicalModel(List<boolean[][]> functionOutputs, int numberOfIterations, List<boolean[]> trueLabels) {
        this.numberOfIterations = numberOfIterations;
        burnInIterations = numberOfIterations / 10;
        numberOfFunctions = functionOutputs.get(0)[0].length;
        numberOfDomains = functionOutputs.size();
        numberOfSamples = new int[numberOfDomains];
        functionOutputsArray = new int[numberOfFunctions][numberOfDomains][];
        for (int p = 0; p < numberOfDomains; p++) {
            numberOfSamples[p] = functionOutputs.get(p).length;
            for (int j = 0; j < numberOfFunctions; j++) {
                functionOutputsArray[j][p] = new int[numberOfSamples[p]];
                for (int i = 0; i < numberOfSamples[p]; i++)
                    functionOutputsArray[j][p][i] = functionOutputs.get(p)[i][j] ? 1 : 0;
            }
        }
        phiSamples = new double[numberOfIterations][numberOfDomains][1 + numberOfFunctions];
        zSamples = new int[numberOfIterations][numberOfDomains];
        labelsSamples = new int[numberOfIterations][numberOfDomains][];
        priorMeans = new double[numberOfDomains];
        priorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
        // Initialization
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfSamples[p]];
            labelVariances[p] = new double[numberOfSamples[p]];
            zSamples[0][p] = p;
            phiSamples[0][p][0] = 0.5;
            for (int j = 0; j < numberOfFunctions; j++)
                phiSamples[0][p][j + 1] = 0.25;
            labelsSamples[0][p] = new int[numberOfSamples[p]];
//            for (int j = 0; j < numberOfIterations; j++) {
//                labelsSamples[j][p] = new int[numberOfSamples[p]];
                for (int i = 0; i < numberOfSamples[p]; i++) {
//                    labelsSamples[j][p][i] = trueLabels.get(p)[i] ? 1 : 0;
                    labelsSamples[0][p][i] = randomDataGenerator.nextBinomial(1, 0.5);
                }
//            }
        }
    }

    public void performGibbsSampling() {
        for (int iterationNumber = 0; iterationNumber < numberOfIterations - 1; iterationNumber++) {
            // Sample phi
            for (int p = 0; p < numberOfDomains; p++) {
                int labelsCount = 0;
                int zCount = 0;
                for (int k = 0; k < numberOfDomains; k++) {
                    if (zSamples[iterationNumber][k] == p) {
                        for (int i = 0; i < numberOfSamples[k]; i++)
                            labelsCount += labelsSamples[iterationNumber][k][i];
                        zCount += numberOfSamples[k];
                    }
                }
                phiSamples[iterationNumber + 1][p][0] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + zCount - labelsCount);
                int numberOfPhiBelowChance = 0;
                for (int j = 0; j < numberOfFunctions; j++) {
                    int disagreementCount = 0;
                    zCount = 0;
                    for (int k = 0; k < numberOfDomains; k++) {
                        if (zSamples[iterationNumber][k] == p) {
                            for (int i = 0; i < numberOfSamples[k]; i++)
                                if (functionOutputsArray[j][k][i] != labelsSamples[iterationNumber][k][i])
                                    disagreementCount++;
                            zCount += numberOfSamples[k];
                        }
                    }
                    phiSamples[iterationNumber + 1][p][j + 1] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + zCount - disagreementCount);
                    if (phiSamples[iterationNumber + 1][p][j + 1] < 0.5)
                        numberOfPhiBelowChance += 1;
                }
                if (numberOfPhiBelowChance < numberOfFunctions / 2.0) {
                    phiSamples[iterationNumber + 1][p][0] = 1 - phiSamples[iterationNumber + 1][p][0];
                    for (int j = 0; j < numberOfFunctions; j++)
                        phiSamples[iterationNumber + 1][p][j + 1] = 1 - phiSamples[iterationNumber + 1][p][j + 1];
                }
            }
            // Sample z
            for (int p = 0; p < numberOfDomains; p++) {
                double[] z_probabilities = new double[numberOfDomains];
                for (int k = 0; k < numberOfDomains; k++)
                    if (k != p) {
                        if (k < p)
                            z_probabilities[zSamples[iterationNumber + 1][k]] += 1;
                        else
                            z_probabilities[zSamples[iterationNumber][k]] += 1;
                    }
                for (int k = 0; k < numberOfDomains; k++)
                    if (z_probabilities[k] == 0.0) {
                        z_probabilities[k] = alpha;
                        break;
                    }
                for (int k = 0; k < numberOfDomains; k++) {
                    z_probabilities[k] = Math.log(z_probabilities[k]);
                    z_probabilities[k] -= Math.log(numberOfDomains - 1 + alpha);
                }
                int count = 0;
                for (int i = 0; i < numberOfSamples[p]; i++)
                    count += labelsSamples[iterationNumber][p][i];
                for (int k = 0; k < numberOfDomains; k++) {
                    if (k < p) {
                        z_probabilities[k] += count * Math.log(phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][k]][0]);
                        z_probabilities[k] += (numberOfSamples[p] - count) * Math.log(1 - phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][k]][0]);
                    } else {
                        z_probabilities[k] += count * Math.log(phiSamples[iterationNumber + 1][zSamples[iterationNumber][k]][0]);
                        z_probabilities[k] += (numberOfSamples[p] - count) * Math.log(1 - phiSamples[iterationNumber + 1][zSamples[iterationNumber][k]][0]);
                    }
                }
                for (int j = 0; j < numberOfFunctions; j++) {
                    count = 0;
                    for (int i = 0; i < numberOfSamples[p]; i++)
                        if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                            count += 1;
                    for (int k = 0; k < numberOfDomains; k++) {
                        if (k < p) {
                            z_probabilities[k] += count * Math.log(phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][k]][j + 1]);
                            z_probabilities[k] += (numberOfSamples[p] - count) * Math.log(1 - phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][k]][j + 1]);
                        } else {
                            z_probabilities[k] += count * Math.log(phiSamples[iterationNumber + 1][zSamples[iterationNumber][k]][j + 1]);
                            z_probabilities[k] += (numberOfSamples[p] - count) * Math.log(1 - phiSamples[iterationNumber + 1][zSamples[iterationNumber][k]][j + 1]);
                        }
                    }
                }
                double normalizationConstant = MatrixUtilities.computeLogSumExp(z_probabilities);
                for (int k = 0; k < numberOfDomains; k++)
                    z_probabilities[k] = Math.exp(z_probabilities[k] - normalizationConstant);
                // Sample from a multinomial
                double[] z_cdf = new double[z_probabilities.length];
                z_cdf[0] = z_probabilities[0];
                for (int i = 1; i < z_probabilities.length; i++)
                    z_cdf[i] = z_cdf[i - 1] + z_probabilities[i];
                double uniform = random.nextDouble();
                zSamples[iterationNumber + 1][p] = numberOfDomains - 1;
                for (int k = 0; k < numberOfDomains; k++) {
                    if (z_cdf[k] > uniform) {
                        zSamples[iterationNumber + 1][p] = k;
                        break;
                    }
                }
//                zSamples[iterationNumber + 1][p] = Arrays.binarySearch(z_cdf, uniform);
//                if (zSamples[iterationNumber + 1][p] < 0)
//                    zSamples[iterationNumber + 1][p] = -zSamples[iterationNumber + 1][p] - 2;
            }
            // Sample labels
            for (int p = 0; p < numberOfDomains; p++) {
                labelsSamples[iterationNumber + 1][p] = new int[numberOfSamples[p]];
                for (int i = 0; i < numberOfSamples[p]; i++) {
                    double p0 = 1 - phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][0];
                    double p1 = phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][0];
                    for (int j = 0; j < numberOfFunctions; j++) {
                        if (functionOutputsArray[j][p][i] == 0) {
                            p0 *= (1 - phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j + 1]);
                            p1 *= phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j + 1];
                        } else {
                            p0 *= phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j + 1];
                            p1 *= (1 - phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j + 1]);
                        }
                    }
                    labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                }
            }
            // Aggregate values for means and variances computation
            if (iterationNumber > burnInIterations)
                for (int p = 0; p < numberOfDomains; p++) {
                    priorMeans[p] += phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][0];
                    for (int j = 0; j < numberOfFunctions; j++)
                        errorRateMeans[p][j] += phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j + 1];
                    for (int i = 0; i < numberOfSamples[p]; i++)
                        labelMeans[p][i] += labelsSamples[iterationNumber + 1][p][i];
                }
        }
        // Compute values for the means and the variances
        for (int p = 0; p < numberOfDomains; p++) {
            priorMeans[p] /= (numberOfIterations - burnInIterations);
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateMeans[p][j] /= (numberOfIterations - burnInIterations);
            for (int i = 0; i < numberOfSamples[p]; i++)
                labelMeans[p][i] /= (numberOfIterations - burnInIterations);
            for (int iterationNumber = burnInIterations; iterationNumber < numberOfIterations - 1; iterationNumber++) {
                double temp = phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][0] - priorMeans[p];
                priorVariances[p] += temp * temp;
                for (int j = 0; j < numberOfFunctions; j++) {
                    temp = phiSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p]][j + 1] - errorRateMeans[p][j];
                    errorRateVariances[p][j] += temp * temp;
                }
                for (int i = 0; i < numberOfSamples[p]; i++) {
                    temp = labelsSamples[iterationNumber + 1][p][i] - labelMeans[p][i];
                    labelVariances[p][i] += temp * temp;
                }
            }
            priorVariances[p] /= (numberOfIterations - burnInIterations - 1);
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateVariances[p][j] /= (numberOfIterations - burnInIterations - 1);
            for (int i = 0; i < numberOfSamples[p]; i++)
                labelVariances[p][i] /= (numberOfIterations - burnInIterations - 1);
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
