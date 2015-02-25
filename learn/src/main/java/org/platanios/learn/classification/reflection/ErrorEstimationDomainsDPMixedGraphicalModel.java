package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;
import org.platanios.learn.math.matrix.MatrixUtilities;

import java.util.*;

import static org.apache.commons.math3.special.Beta.logBeta;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationDomainsDPMixedGraphicalModel {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double alpha = 100;
    private final double alpha_p = 1;
    private final double beta_p = 1;
    private final double alpha_e = 1;
    private final double beta_e = 100;

    private final int numberOfIterations;
    private final int burnInIterations;
    private final int thinning = 10;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfDataSamples;
    private final int[][][] labelsSamples;
    private final int[][][] functionOutputsArray;
    private final int[][][] zSamples;
    private final double[][] priorSamples;
    private final double[][] errorRateSamples;

    private double[][] disagreements;
    private double[][] sum_1;
    private double[][] sum_2;

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;

    public ErrorEstimationDomainsDPMixedGraphicalModel(List<boolean[][]> functionOutputs, int numberOfIterations, List<boolean[]> trueLabels) {
        this.numberOfIterations = numberOfIterations;
        burnInIterations = numberOfIterations * 9 / 10;
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
        numberOfSamples = (numberOfIterations - burnInIterations) / thinning;
        priorSamples = new double[numberOfSamples][numberOfDomains];
        errorRateSamples = new double[numberOfSamples][numberOfDomains * numberOfFunctions];
        zSamples = new int[numberOfSamples][numberOfDomains][numberOfFunctions];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];

        disagreements = new double[numberOfFunctions][numberOfDomains];
        sum_1 = new double[numberOfFunctions][numberOfDomains * numberOfFunctions];
        sum_2 = new double[numberOfFunctions][numberOfDomains * numberOfFunctions];

        priorMeans = new double[numberOfDomains];
        priorVariances = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        labelVariances = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        errorRateVariances = new double[numberOfDomains][numberOfFunctions];
        for (int p = 0; p < numberOfDomains; p++) {
            labelMeans[p] = new double[numberOfDataSamples[p]];
            labelVariances[p] = new double[numberOfDataSamples[p]];
            priorSamples[0][p] = 0.5;
            labelsSamples[0][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsSamples[0][p][i] = randomDataGenerator.nextBinomial(1, 0.5);
            for (int j = 0; j < numberOfFunctions; j++) {
                zSamples[0][p][j] = 0;
                errorRateSamples[0][p * numberOfFunctions + j] = 0.25;
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[0][p][i])
                        disagreements[j][p]++;
            }
        }
        for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                sum_1[j][k] = 0;
                sum_2[j][k] = 0;
                for (int p = 0; p < numberOfDomains; p++) {
                    if (zSamples[0][p][j] == k) {
                        sum_1[j][k] += numberOfDataSamples[p];
                        sum_2[j][k] += disagreements[j][p];
                    }
                }
            }
        }
    }

    public void performGibbsSampling() {
        for (int iterationNumber = 0; iterationNumber < burnInIterations; iterationNumber++) {
            samplePriorsAndBurn(0);
            sampleLabelsAndBurn(0);
            sampleZAndBurnWithCollapsedErrorRates(0);
//            samplePriorsAndBurn(0);
//            sampleErrorRatesAndBurn(0);
//            sampleZAndBurn(0);
//            sampleLabelsAndBurn(0);
            if (iterationNumber % 100 == 0)
                System.out.println("Iteration #" + iterationNumber);
        }
        for (int iterationNumber = 0; iterationNumber < numberOfSamples - 1; iterationNumber++) {
            for (int i = 0; i < thinning; i++) {
                samplePriorsAndBurn(iterationNumber);
                sampleErrorRatesAndBurn(iterationNumber);
                sampleZAndBurn(iterationNumber);
                sampleLabelsAndBurn(iterationNumber);
            }
            samplePriors(iterationNumber);
            sampleErrorRates(iterationNumber);
            sampleZ(iterationNumber);
            sampleLabels(iterationNumber);
        }
        Set<Integer> uniqueClusters = new HashSet<>();
        for (int p = 0; p < numberOfDomains; p++)
            for (int j = 0; j < numberOfFunctions; j++)
                uniqueClusters.add(zSamples[zSamples.length - 1][p][j]);
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                int numberOfPhiBelowChance = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    if (errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]] < 0.5)
                        numberOfPhiBelowChance++;
                if (numberOfPhiBelowChance < numberOfFunctions / 2.0) {
                    priorSamples[sampleNumber][p] = 1 - priorSamples[sampleNumber][p];
                    for (int j = 0; j < numberOfFunctions; j++)
                        errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]] = 1 - errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]];
                }
                priorMeans[p] += priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]];
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
                    temp = errorRateSamples[sampleNumber][zSamples[sampleNumber][p][j]] - errorRateMeans[p][j];
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
            priorSamples[iterationNumber][p] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + numberOfDataSamples[p] - labelsCount);
        }
    }

    private void samplePriors(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[iterationNumber][p][i];
            priorSamples[iterationNumber + 1][p] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + numberOfDataSamples[p] - labelsCount);
        }
    }

    private void sampleErrorRatesAndBurn(int iterationNumber) {
        for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
            int disagreementCount = 0;
            int zCount = 0;
            for (int p = 0; p < numberOfDomains; p++) {
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (zSamples[iterationNumber][p][j] == k) {
                        for (int i = 0; i < numberOfDataSamples[p]; i++)
                            if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[p];
                    }
                }
            }
            if (zCount > 0)
                errorRateSamples[iterationNumber][k] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + zCount - disagreementCount);
        }
    }

    private void sampleErrorRates(int iterationNumber) {
        for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
            int disagreementCount = 0;
            int zCount = 0;
            for (int p = 0; p < numberOfDomains; p++) {
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (zSamples[iterationNumber][p][j] == k) {
                        for (int i = 0; i < numberOfDataSamples[p]; i++)
                            if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                                disagreementCount++;
                        zCount += numberOfDataSamples[p];
                    }
                }
            }
            if (zCount > 0)
                errorRateSamples[iterationNumber + 1][k] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + zCount - disagreementCount);
        }
    }

    private void sampleZAndBurnWithCollapsedErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
            }
        }
        for (int j = 0; j < numberOfFunctions; j++) {
            for (int k = 0; k < numberOfDomains * numberOfFunctions; k++) {
                sum_2[j][k] = 0;
                for (int p = 0; p < numberOfDomains; p++) {
                    if (zSamples[iterationNumber][p][j] == k)
                        sum_2[j][k] += disagreements[j][p];
                }
            }
        }
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                double[] z_probabilities = new double[numberOfDomains * numberOfFunctions];
                for (int p_inner = 0; p_inner < numberOfDomains; p_inner++)
                    for (int j_inner = 0; j_inner < numberOfFunctions; j_inner++)
                        if (p_inner != p || j_inner != j)
                            z_probabilities[zSamples[iterationNumber][p_inner][j_inner]]++;
                int k_new = -1;
                for (int k = 0; k < z_probabilities.length; k++)
                    if (z_probabilities[k] == 0.0) {
                        z_probabilities[k] = alpha;
                        k_new = k;
                        break;
                    }
                for (int k = 0; k < z_probabilities.length; k++) {
                    z_probabilities[k] = Math.log(z_probabilities[k]);
                    z_probabilities[k] -= Math.log(numberOfDomains - 1 + alpha);
                }
                for (int k = 0; k < z_probabilities.length; k++) {
                    if (zSamples[iterationNumber][p][j] == k) {
                        sum_1[j][k] -= numberOfDataSamples[p];
                        sum_2[j][k] -= disagreements[j][p];
                    }
                    if (k != k_new) {
                        double alpha = alpha_e + sum_2[j][k] + disagreements[j][p];
                        double beta = beta_e + sum_1[j][k] - sum_2[j][k] + numberOfDataSamples[p] - disagreements[j][p];
                        z_probabilities[k] += logBeta(alpha, beta) - logBeta(alpha_e + sum_2[j][k], beta_e + sum_1[j][k] - sum_2[j][k]);
                    } else {
                        z_probabilities[k_new] += logBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]) - logBeta(alpha_e, beta_e);
                    }
                }
                double normalizationConstant = MatrixUtilities.computeLogSumExp(z_probabilities);
                for (int k = 0; k < z_probabilities.length; k++)
                    z_probabilities[k] = Math.exp(z_probabilities[k] - normalizationConstant);
                // Sample from a multinomial
                double[] z_cdf = new double[z_probabilities.length];
                z_cdf[0] = z_probabilities[0];
                for (int i = 1; i < z_probabilities.length; i++)
                    z_cdf[i] = z_cdf[i - 1] + z_probabilities[i];
                double uniform = random.nextDouble();
                zSamples[iterationNumber][p][j] = z_probabilities.length - 1;
                for (int k = 0; k < z_probabilities.length - 1; k++) {
                    if (z_cdf[k] > uniform) {
                        zSamples[iterationNumber][p][j] = k;
                        sum_1[j][k] += numberOfDataSamples[p];
                        sum_2[j][k] += disagreements[j][p];
                        break;
                    }
                }
                if (zSamples[iterationNumber][p][j] == z_probabilities.length - 1) {
                    sum_1[j][z_probabilities.length - 1] += numberOfDataSamples[p];
                    sum_2[j][z_probabilities.length - 1] += disagreements[j][p];
                }
            }
        }
    }

    private void sampleZAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                double[] z_probabilities = new double[numberOfDomains * numberOfFunctions];
                for (int p_inner = 0; p_inner < numberOfDomains; p_inner++)
                    for (int j_inner = 0; j_inner < numberOfFunctions; j_inner++)
                        if (p_inner != p || j_inner != j)
                            z_probabilities[zSamples[iterationNumber][p_inner][j_inner]] += 1;
                int k_new = -1;
                for (int k = 0; k < z_probabilities.length; k++)
                    if (z_probabilities[k] == 0.0) {
                        z_probabilities[k] = alpha;
                        k_new = k;
                        break;
                    }
                for (int k = 0; k < z_probabilities.length; k++) {
                    z_probabilities[k] = Math.log(z_probabilities[k]);
                    z_probabilities[k] -= Math.log(numberOfDomains - 1 + alpha);
                }
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
                for (int k = 0; k < z_probabilities.length; k++) {
                    if (k != k_new) {
                        z_probabilities[k] += disagreements[j][p] * Math.log(errorRateSamples[iterationNumber][k]);
                        z_probabilities[k] += (numberOfDataSamples[p] - disagreements[j][p]) * Math.log(1 - errorRateSamples[iterationNumber][k]);
                    } else {
                        z_probabilities[k_new] += logBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]) - logBeta(alpha_e, beta_e);
                    }
                }
                double normalizationConstant = MatrixUtilities.computeLogSumExp(z_probabilities);
                for (int k = 0; k < z_probabilities.length; k++)
                    z_probabilities[k] = Math.exp(z_probabilities[k] - normalizationConstant);
                // Sample from a multinomial
                double[] z_cdf = new double[z_probabilities.length];
                z_cdf[0] = z_probabilities[0];
                for (int i = 1; i < z_probabilities.length; i++)
                    z_cdf[i] = z_cdf[i - 1] + z_probabilities[i];
                double uniform = random.nextDouble();
                zSamples[iterationNumber][p][j] = z_probabilities.length - 1;
                for (int k = 0; k < z_probabilities.length - 1; k++) {
                    if (z_cdf[k] > uniform) {
                        zSamples[iterationNumber][p][j] = k;
                        break;
                    }
                }
            }
        }
    }

    private void sampleZ(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int j = 0; j < numberOfFunctions; j++) {
                double[] z_probabilities = new double[numberOfDomains * numberOfFunctions];
                for (int p_inner = 0; p_inner < numberOfDomains; p_inner++)
                    for (int j_inner = 0; j_inner < numberOfFunctions; j_inner++)
                        if (p_inner < p || (p_inner == p && j_inner < p))
                            z_probabilities[zSamples[iterationNumber + 1][p_inner][j_inner]] += 1;
                        else if (p_inner != p || j_inner != j)
                            z_probabilities[zSamples[iterationNumber][p_inner][j_inner]] += 1;
                int k_new = -1;
                for (int k = 0; k < z_probabilities.length; k++)
                    if (z_probabilities[k] == 0.0) {
                        z_probabilities[k] = alpha;
                        k_new = k;
                        break;
                    }
                for (int k = 0; k < z_probabilities.length; k++) {
                    z_probabilities[k] = Math.log(z_probabilities[k]);
                    z_probabilities[k] -= Math.log(numberOfDomains - 1 + alpha);
                }
                disagreements[j][p] = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreements[j][p]++;
                for (int k = 0; k < z_probabilities.length; k++) {
                    if (k != k_new) {
                        z_probabilities[k] += disagreements[j][p] * Math.log(errorRateSamples[iterationNumber + 1][k]);
                        z_probabilities[k] += (numberOfDataSamples[p] - disagreements[j][p]) * Math.log(1 - errorRateSamples[iterationNumber + 1][k]);
                    } else {
                        z_probabilities[k_new] += logBeta(alpha_e + disagreements[j][p], beta_e + numberOfDataSamples[p] - disagreements[j][p]) - logBeta(alpha_e, beta_e);
                    }
                }
                double normalizationConstant = MatrixUtilities.computeLogSumExp(z_probabilities);
                for (int k = 0; k < z_probabilities.length; k++)
                    z_probabilities[k] = Math.exp(z_probabilities[k] - normalizationConstant);
                // Sample from a multinomial
                double[] z_cdf = new double[z_probabilities.length];
                z_cdf[0] = z_probabilities[0];
                for (int i = 1; i < z_probabilities.length; i++)
                    z_cdf[i] = z_cdf[i - 1] + z_probabilities[i];
                double uniform = random.nextDouble();
                zSamples[iterationNumber + 1][p][j] = z_probabilities.length - 1;
                for (int k = 0; k < z_probabilities.length - 1; k++) {
                    if (z_cdf[k] > uniform) {
                        zSamples[iterationNumber + 1][p][j] = k;
                        break;
                    }
                }
            }
        }
    }

    private void sampleLabelsAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[iterationNumber][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSamples[iterationNumber][p];
                double p1 = priorSamples[iterationNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]]);
                        p1 *= errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]];
                    } else {
                        p0 *= errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]];
                        p1 *= (1 - errorRateSamples[iterationNumber][zSamples[iterationNumber][p][j]]);
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
                double p0 = 1 - priorSamples[iterationNumber + 1][p];
                double p1 = priorSamples[iterationNumber + 1][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p][j]]);
                        p1 *= errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p][j]];
                    } else {
                        p0 *= errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p][j]];
                        p1 *= (1 - errorRateSamples[iterationNumber + 1][zSamples[iterationNumber + 1][p][j]]);
                    }
                }
                labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
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
