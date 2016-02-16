package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BayesianErrorEstimationConfusion {
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double labelsPriorAlpha = 1;
    private final double labelsPriorBeta = 1;
    private final double[][] confusionMatrixPrior = new double[][] { new double[] { 10, 1 }, new double[] { 1, 10 } };

    private final int numberOfBurnInSamples;
    private final int numberOfThinningSamples;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfDataSamples;
    private final int[][][] functionOutputsArray;
    private final double[][] labelPriorsSamples;
    private final int[][][] labelsSamples;
    private final double[][][][][] confusionMatrixSamples;
    private final double[][][][] confusionMatrixCounts;

    private final double[] labelPriorMeans;
    private final double[][] labelMeans;
    private final double[][] errorRateMeans;

    public BayesianErrorEstimationConfusion(List<boolean[][]> functionOutputs,
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
        labelPriorsSamples = new double[numberOfSamples][numberOfDomains];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];
        confusionMatrixSamples = new double[numberOfSamples][numberOfDomains][numberOfFunctions][2][2];
        confusionMatrixCounts = new double[numberOfDomains][numberOfFunctions][2][2];
        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
            for (int p = 0; p < numberOfDomains; p++)
                labelsSamples[sampleIndex][p] = new int[numberOfDataSamples[p]];
        for (int p = 0; p < numberOfDomains; p++) {
            labelPriorsSamples[0][p] = 0.5;
            for (int j = 0; j < numberOfFunctions; j++) {
                confusionMatrixSamples[0][p][j][0][0] = 0.75;
                confusionMatrixSamples[0][p][j][0][1] = 0.25;
                confusionMatrixSamples[0][p][j][1][0] = 0.25;
                confusionMatrixSamples[0][p][j][1][1] = 0.75;
            }
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
                updateCountsAfterSamplingLabel(0, p, i);
            }
        }
        labelPriorMeans = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
    }

    public void runGibbsSampler() {
        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
            samplePriorsAndConfusionMatrix(0);
            sampleLabels(0);
        }
        for (int sampleIndex = 1; sampleIndex < numberOfSamples; sampleIndex++) {
            for (int i = 0; i < numberOfThinningSamples; i++) {
                samplePriorsAndConfusionMatrix(sampleIndex - 1);
                sampleLabels(sampleIndex - 1);
            }
            samplePriorsAndConfusionMatrix(sampleIndex - 1);
            sampleLabels(sampleIndex - 1);
            storeSample(sampleIndex);
        }
        // Aggregate values for means and variances computation
        for (int p = 0; p < numberOfDomains; p++)
            labelMeans[p] = new double[numberOfDataSamples[p]];
        // Aggregate values for means and variances computation
        for (int sampleNumber = 0; sampleNumber < numberOfSamples; sampleNumber++) {
            for (int p = 0; p < numberOfDomains; p++) {
                labelPriorMeans[p] += labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    double errorRate = 0;
                    for (int i = 0; i < numberOfDataSamples[p]; i++)
                        errorRate += functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i] ? 1 : 0;
                    errorRateMeans[p][j] += errorRate / numberOfDataSamples[p];
                }
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    labelMeans[p][i] += labelsSamples[sampleNumber][p][i];
            }
        }
        // Compute values for the means and the variances
        for (int p = 0; p < numberOfDomains; p++) {
            labelPriorMeans[p] /= numberOfSamples;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRateMeans[p][j] /= numberOfSamples;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelMeans[p][i] /= numberOfSamples;
        }
    }

    private void updateCountsBeforeSamplingLabel(int sampleNumber, int p, int i) {
        for (int j = 0; j < numberOfFunctions; j++)
            confusionMatrixCounts[p][j][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]--;
    }

    private void updateCountsAfterSamplingLabel(int sampleNumber, int p, int i) {
        for (int j = 0; j < numberOfFunctions; j++)
            confusionMatrixCounts[p][j][labelsSamples[sampleNumber][p][i]][functionOutputsArray[j][p][i]]++;
    }

    private void samplePriorsAndConfusionMatrix(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[sampleNumber][p][i];
            labelPriorsSamples[sampleNumber][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelsCount, labelsPriorBeta + numberOfDataSamples[p] - labelsCount);
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                confusionMatrixSamples[sampleNumber][p][j][0][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[0][0] + confusionMatrixCounts[p][j][0][0],
                                                                                             confusionMatrixPrior[0][1] + confusionMatrixCounts[p][j][0][1]);
                confusionMatrixSamples[sampleNumber][p][j][1][0] = randomDataGenerator.nextBeta(confusionMatrixPrior[1][0] + confusionMatrixCounts[p][j][1][0],
                                                                                             confusionMatrixPrior[1][1] + confusionMatrixCounts[p][j][1][1]);
                confusionMatrixSamples[sampleNumber][p][j][0][1] = 1 - confusionMatrixSamples[sampleNumber][p][j][0][0];
                confusionMatrixSamples[sampleNumber][p][j][1][1] = 1 - confusionMatrixSamples[sampleNumber][p][j][1][0];
            }
        }
    }

    private void sampleLabels(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - labelPriorsSamples[sampleNumber][p]; // TODO: Compute this in log-space
                double p1 = labelPriorsSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    p0 *= confusionMatrixSamples[sampleNumber][p][j][0][functionOutputsArray[j][p][i]];
                    p1 *= confusionMatrixSamples[sampleNumber][p][j][1][functionOutputsArray[j][p][i]];
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

    private void storeSample(int sampleIndex) {
        copyArray(labelPriorsSamples[sampleIndex - 1], labelPriorsSamples[sampleIndex]);
        copyArray(labelsSamples[sampleIndex - 1], labelsSamples[sampleIndex]);
        copyArray(confusionMatrixSamples[sampleIndex - 1], confusionMatrixSamples[sampleIndex]);
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

    public double[] getLabelPriorMeans() {
        return labelPriorMeans;
    }

    public double[][] getLabelPriorsSamples() {
        return labelPriorsSamples;
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

    public double[][][][][] getConfusionMatrixSamples() {
        return confusionMatrixSamples;
    }
}
