package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationGraphicalModel {
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double alpha_p = 1;
    private final double beta_p = 1;
    private final double alpha_e = 1;
    private final double beta_e = 100;

    private final int numberOfBurnInSamples;
    private final int numberOfThinningSamples;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfDataSamples;
    private final int[][][] functionOutputsArray;
    private final double[] priorSample;
    private final int[][] labelsSample;
    private final double[][] errorRatesSample;
    private final double[][] priorSamples;
    private final int[][][] labelsSamples;
    private final double[][][] errorRatesSamples;

    private double[] priorMeans;
    private double[][] labelMeans;
    private double[][] errorRateMeans;

    public ErrorEstimationGraphicalModel(List<boolean[][]> functionOutputs,
                                         int numberOfBurnInSamples,
                                         int numberOfThinningSamples,
                                         int numberOfSamples) {
        this.numberOfBurnInSamples = numberOfBurnInSamples;
        this.numberOfThinningSamples = numberOfThinningSamples;
        this.numberOfSamples = numberOfSamples;
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
        priorSample = new double[numberOfDomains];
        labelsSample = new int[numberOfDomains][];
        errorRatesSample = new double[numberOfDomains][numberOfFunctions];
        for (int p = 0; p < numberOfDomains; p++) {
            priorSample[p] = 0.5;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRatesSample[p][j] = 0.25;
            labelsSample[p] = new int[numberOfDataSamples[p]];
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSample[p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
            }
        }
        priorSamples = new double[this.numberOfSamples][numberOfDomains];
        labelsSamples = new int[this.numberOfSamples][numberOfDomains][];
        errorRatesSamples = new double[this.numberOfSamples][numberOfDomains][numberOfFunctions];
        priorMeans = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        for (int p = 0; p < numberOfDomains; p++)
            labelMeans[p] = new double[numberOfDataSamples[p]];
    }

    public void runGibbsSampler() {
        for (int sample = 0; sample < numberOfBurnInSamples; sample++) {
            samplePriorsAndErrorRates();
            sampleLabels();
        }
        for (int sample = 0; sample < numberOfSamples; sample++) {
            samplePriorsAndErrorRates();
            sampleLabels();
            storeSample(sample);
            for (int i = 0; i < numberOfThinningSamples; i++) {
                samplePriorsAndErrorRates();
                sampleLabels();
            }
        }
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                int numberOfPhiBelowChance = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    if (errorRatesSamples[sampleNumber][p][j] < 0.5)
                        numberOfPhiBelowChance++;
                if (numberOfPhiBelowChance < numberOfFunctions / 2.0) {
                    priorSamples[sampleNumber][p] = 1 - priorSamples[sampleNumber][p];
                    for (int j = 0; j < numberOfFunctions; j++)
                        errorRatesSamples[sampleNumber][p][j] = 1 - errorRatesSamples[sampleNumber][p][j];
                }
                priorMeans[p] += priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRateMeans[p][j] += errorRatesSamples[sampleNumber][p][j];
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
        }
    }

    private void samplePriorsAndErrorRates() {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSample[p][i];
            priorSample[p] = randomDataGenerator.nextBeta(alpha_p + labelsCount,
                                                          beta_p + numberOfDataSamples[p] - labelsCount);
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSample[p][i])
                        disagreementCount++;
                errorRatesSample[p][j] =
                        randomDataGenerator.nextBeta(alpha_e + disagreementCount,
                                                     beta_e + numberOfDataSamples[p] - disagreementCount);
                if (errorRatesSample[p][j] < 0.5)
                    numberOfErrorRatesBelowChance += 1;
            }
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0) {
                priorSample[p] = 1 - priorSample[p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRatesSample[p][j] = 1 - errorRatesSample[p][j];
            }
        }
    }

    private void sampleLabels() {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSample[p];
                double p1 = priorSample[p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRatesSample[p][j]);
                        p1 *=errorRatesSample[p][j];
                    } else {
                        p0 *= errorRatesSample[p][j];
                        p1 *= (1 - errorRatesSample[p][j]);
                    }
                }
                labelsSample[p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
            }
        }
    }

    private void storeSample(int sample) {
        for (int p = 0; p < numberOfDomains; p++) {
            priorSamples[sample][p] = priorSample[p];
            labelsSamples[sample][p] = new int[numberOfDataSamples[p]];
            System.arraycopy(labelsSample[p], 0, labelsSamples[sample][p], 0, numberOfDataSamples[p]);
            System.arraycopy(errorRatesSample[p], 0, errorRatesSamples[sample][p], 0, numberOfFunctions);
        }
    }

    public double[] getPriorMeans() {
        return priorMeans;
    }

    public double[][] getPriorSamples() {
        return priorSamples;
    }

    public double[][] getLabelMeans() {
        return labelMeans;
    }

    public int[][][] getLabelSamples() {
        return labelsSamples;
    }

    public double[][] getErrorRatesMeans() {
        return errorRateMeans;
    }

    public double[][][] getErrorRatesSamples() {
        return errorRatesSamples;
    }
}
