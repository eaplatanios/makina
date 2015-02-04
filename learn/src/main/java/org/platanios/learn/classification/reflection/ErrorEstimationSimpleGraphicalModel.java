package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.List;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationSimpleGraphicalModel {
    private final Random random = new Random();
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
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
    private final double[][] priorSamples;
    private final double[][][] errorRateSamples;

    private double[] priorMeans;
    private double[] priorVariances;
    private double[][] labelMeans;
    private double[][] labelVariances;
    private double[][] errorRateMeans;
    private double[][] errorRateVariances;

    public ErrorEstimationSimpleGraphicalModel(List<boolean[][]> functionOutputs, int numberOfIterations) {
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
        priorSamples = new double[numberOfIterations][numberOfDomains];
        errorRateSamples = new double[numberOfIterations][numberOfDomains][numberOfFunctions];
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
            priorSamples[0][p] = 0.5;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateSamples[0][p][j] = 0.25;
            labelsSamples[0][p] = new int[numberOfSamples[p]];
            for (int i = 0; i < numberOfSamples[p]; i++)
                labelsSamples[0][p][i] = randomDataGenerator.nextBinomial(1, 0.5);
        }
    }

    public void performGibbsSampling() {
        for (int iterationNumber = 0; iterationNumber < numberOfIterations - 1; iterationNumber++) {
            // Sample priors and error rates
            for (int p = 0; p < numberOfDomains; p++) {
                int labelsCount = 0;
                for (int i = 0; i < numberOfSamples[p]; i++)
                    labelsCount += labelsSamples[iterationNumber][p][i];
                priorSamples[iterationNumber + 1][p] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + numberOfSamples[p] - labelsCount);
                int numberOfErrorRatesBelowChance = 0;
                for (int j = 0; j < numberOfFunctions; j++) {
                    int disagreementCount = 0;
                    for (int i = 0; i < numberOfSamples[p]; i++)
                        if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                            disagreementCount++;
                    errorRateSamples[iterationNumber + 1][p][j] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + numberOfSamples[p] - disagreementCount);
                    if (errorRateSamples[iterationNumber + 1][p][j] < 0.5)
                        numberOfErrorRatesBelowChance += 1;
                }
                if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0) {
                    priorSamples[iterationNumber + 1][p] = 1 - priorSamples[iterationNumber + 1][p];
                    for (int j = 0; j < numberOfFunctions; j++)
                        errorRateSamples[iterationNumber + 1][p][j] = 1 - errorRateSamples[iterationNumber + 1][p][j];
                }
            }
            // Sample labels
            for (int p = 0; p < numberOfDomains; p++) {
                labelsSamples[iterationNumber + 1][p] = new int[numberOfSamples[p]];
                for (int i = 0; i < numberOfSamples[p]; i++) {
                    double p0 = 1 - priorSamples[iterationNumber + 1][p];
                    double p1 = priorSamples[iterationNumber + 1][p];
                    for (int j = 0; j < numberOfFunctions; j++) {
                        if (functionOutputsArray[j][p][i] == 0) {
                            p0 *= (1 - errorRateSamples[iterationNumber + 1][p][j]);
                            p1 *=errorRateSamples[iterationNumber + 1][p][j];
                        } else {
                            p0 *= errorRateSamples[iterationNumber + 1][p][j];
                            p1 *= (1 - errorRateSamples[iterationNumber + 1][p][j]);
                        }
                    }
                    labelsSamples[iterationNumber + 1][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
                }
            }
            // Aggregate values for means and variances computation
            if (iterationNumber > burnInIterations)
                for (int p = 0; p < numberOfDomains; p++) {
                    priorMeans[p] += priorSamples[iterationNumber + 1][p];
                    for (int j = 0; j < numberOfFunctions; j++)
                        errorRateMeans[p][j] += errorRateSamples[iterationNumber + 1][p][j];
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
                double temp = priorSamples[iterationNumber + 1][p] - priorMeans[p];
                priorVariances[p] += temp * temp;
                for (int j = 0; j < numberOfFunctions; j++) {
                    temp = errorRateSamples[iterationNumber + 1][p][j] - errorRateMeans[p][j];
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
