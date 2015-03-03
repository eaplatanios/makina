package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationSimpleGraphicalModel {
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double alpha_p = 1;
    private final double beta_p = 1;
    private final double alpha_e = 1;
    private final double beta_e = 100;

    private final int numberOfIterations;
    private final int burnInIterations;
    private final int thinning;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfDataSamples;
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

    public ErrorEstimationSimpleGraphicalModel(List<boolean[][]> functionOutputs, int numberOfIterations, int thinning) {
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
        numberOfSamples = (numberOfIterations - burnInIterations) / thinning;
        priorSamples = new double[numberOfSamples][numberOfDomains];
        errorRateSamples = new double[numberOfSamples][numberOfDomains][numberOfFunctions];
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
            priorSamples[0][p] = 0.5;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateSamples[0][p][j] = 0.25;
            labelsSamples[0][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
//                labelsSamples[0][p][i] = randomDataGenerator.nextBinomial(1, 0.5);
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
            }
        }
    }

    public void performGibbsSampling() {
        for (int iterationNumber = 0; iterationNumber < burnInIterations; iterationNumber++) {
            samplePriorsAndErrorRatesAndBurn(0);
            sampleLabelsAndBurn(0);
        }
        for (int iterationNumber = 0; iterationNumber < numberOfSamples - 1; iterationNumber++) {
            samplePriorsAndErrorRates(iterationNumber);
            sampleLabels(iterationNumber);
            for (int i = 0; i < thinning; i++) {
                samplePriorsAndErrorRatesAndBurn(iterationNumber);
                sampleLabelsAndBurn(iterationNumber);
            }
        }
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                int numberOfPhiBelowChance = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    if (errorRateSamples[sampleNumber][p][j] < 0.5)
                        numberOfPhiBelowChance++;
                if (numberOfPhiBelowChance < numberOfFunctions / 2.0) {
                    priorSamples[sampleNumber][p] = 1 - priorSamples[sampleNumber][p];
                    for (int j = 0; j < numberOfFunctions; j++)
                        errorRateSamples[sampleNumber][p][j] = 1 - errorRateSamples[sampleNumber][p][j];
                }
                priorMeans[p] += priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRateSamples[sampleNumber][p][j];
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
                    temp = errorRateSamples[sampleNumber][p][j] - errorRateMeans[p][j];
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

    private void samplePriorsAndErrorRatesAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[iterationNumber][p][i];
            priorSamples[iterationNumber][p] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + numberOfDataSamples[p] - labelsCount);
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreementCount++;
                errorRateSamples[iterationNumber][p][j] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + numberOfDataSamples[p] - disagreementCount);
                if (errorRateSamples[iterationNumber][p][j] < 0.5)
                    numberOfErrorRatesBelowChance += 1;
            }
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0) {
                priorSamples[iterationNumber][p] = 1 - priorSamples[iterationNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateSamples[iterationNumber][p][j] = 1 - errorRateSamples[iterationNumber][p][j];
            }
        }
    }

    private void samplePriorsAndErrorRates(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[iterationNumber][p][i];
            priorSamples[iterationNumber + 1][p] = randomDataGenerator.nextBeta(alpha_p + labelsCount, beta_p + numberOfDataSamples[p] - labelsCount);
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[iterationNumber][p][i])
                        disagreementCount++;
                errorRateSamples[iterationNumber + 1][p][j] = randomDataGenerator.nextBeta(alpha_e + disagreementCount, beta_e + numberOfDataSamples[p] - disagreementCount);
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

    private void sampleLabelsAndBurn(int iterationNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            labelsSamples[iterationNumber][p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSamples[iterationNumber][p];
                double p1 = priorSamples[iterationNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRateSamples[iterationNumber][p][j]);
                        p1 *=errorRateSamples[iterationNumber][p][j];
                    } else {
                        p0 *= errorRateSamples[iterationNumber][p][j];
                        p1 *= (1 - errorRateSamples[iterationNumber][p][j]);
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
