package org.platanios.learn.classification.reflection;

import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationGraphicalModel {
    private final RandomDataGenerator randomDataGenerator = new RandomDataGenerator();
    private final double labelsPriorAlpha = 1;
    private final double labelsPriorBeta = 1;
    private final double errorRatesPriorAlpha = 1;
    private final double errorRatesPriorBeta = 10;

    private final int numberOfBurnInSamples;
    private final int numberOfThinningSamples;
    private final int numberOfSamples;
    private final int numberOfFunctions;
    private final int numberOfDomains;
    private final int[] numberOfDataSamples;
    private final int[][][] functionOutputsArray;
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
        priorSamples = new double[numberOfSamples][numberOfDomains];
        labelsSamples = new int[numberOfSamples][numberOfDomains][];
        errorRatesSamples = new double[numberOfSamples][numberOfDomains][numberOfFunctions];
        for (int sampleIndex = 0; sampleIndex < numberOfSamples; sampleIndex++)
            for (int p = 0; p < numberOfDomains; p++)
                labelsSamples[sampleIndex][p] = new int[numberOfDataSamples[p]];
        for (int p = 0; p < numberOfDomains; p++) {
            priorSamples[0][p] = 0.5;
            for (int j = 0; j < numberOfFunctions; j++)
                errorRatesSamples[0][p][j] = 0.25;
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                int sum = 0;
                for (int j = 0; j < numberOfFunctions; j++)
                    sum += functionOutputsArray[j][p][i];
                labelsSamples[0][p][i] = sum >= (numberOfFunctions / 2) ? 1 : 0;
            }
        }
        priorMeans = new double[numberOfDomains];
        labelMeans = new double[numberOfDomains][];
        errorRateMeans = new double[numberOfDomains][numberOfFunctions];
        for (int p = 0; p < numberOfDomains; p++)
            labelMeans[p] = new double[numberOfDataSamples[p]];
    }

    public void runGibbsSampler() {
        for (int sampleIndex = 0; sampleIndex < numberOfBurnInSamples; sampleIndex++) {
            samplePriorsAndErrorRates(0);
            sampleLabels(0);
        }
        for (int sampleIndex = 1; sampleIndex < numberOfSamples; sampleIndex++) {
            for (int i = 0; i < numberOfThinningSamples; i++) {
                samplePriorsAndErrorRates(sampleIndex - 1);
                sampleLabels(sampleIndex - 1);
            }
            samplePriorsAndErrorRates(sampleIndex - 1, true);
            sampleLabels(sampleIndex - 1);
            storeSample(sampleIndex);
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

    private void samplePriorsAndErrorRates(int sampleNumber) {
        samplePriorsAndErrorRates(sampleNumber, false);
    }

    private void samplePriorsAndErrorRates(int sampleNumber, boolean sampleMean) {
        for (int p = 0; p < numberOfDomains; p++) {
            int labelsCount = 0;
            for (int i = 0; i < numberOfDataSamples[p]; i++)
                labelsCount += labelsSamples[sampleNumber][p][i];
            if (sampleMean)
                priorSamples[sampleNumber][p] = (labelsPriorAlpha + labelsCount) / (labelsPriorAlpha + labelsPriorBeta + numberOfDataSamples[p]);
            else
                priorSamples[sampleNumber][p] = randomDataGenerator.nextBeta(labelsPriorAlpha + labelsCount, labelsPriorBeta + numberOfDataSamples[p] - labelsCount);
            int numberOfErrorRatesBelowChance = 0;
            for (int j = 0; j < numberOfFunctions; j++) {
                int disagreementCount = 0;
                for (int i = 0; i < numberOfDataSamples[p]; i++)
                    if (functionOutputsArray[j][p][i] != labelsSamples[sampleNumber][p][i])
                        disagreementCount++;
                if (sampleMean)
                    errorRatesSamples[sampleNumber][p][j] = (errorRatesPriorAlpha + disagreementCount) / (errorRatesPriorAlpha + errorRatesPriorBeta + numberOfDataSamples[p]);
                else
                    errorRatesSamples[sampleNumber][p][j] =
                            randomDataGenerator.nextBeta(errorRatesPriorAlpha + disagreementCount, errorRatesPriorBeta + numberOfDataSamples[p] - disagreementCount);
                if (errorRatesSamples[sampleNumber][p][j] < 0.5)
                    numberOfErrorRatesBelowChance += 1;
            }
            if (numberOfErrorRatesBelowChance < numberOfFunctions / 2.0) {
                priorSamples[sampleNumber][p] = 1 - priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++)
                    errorRatesSamples[sampleNumber][p][j] = 1 - errorRatesSamples[sampleNumber][p][j];
            }
        }
    }

    private void sampleLabels(int sampleNumber) {
        for (int p = 0; p < numberOfDomains; p++) {
            for (int i = 0; i < numberOfDataSamples[p]; i++) {
                double p0 = 1 - priorSamples[sampleNumber][p];
                double p1 = priorSamples[sampleNumber][p];
                for (int j = 0; j < numberOfFunctions; j++) {
                    if (functionOutputsArray[j][p][i] == 0) {
                        p0 *= (1 - errorRatesSamples[sampleNumber][p][j]);
                        p1 *=errorRatesSamples[sampleNumber][p][j];
                    } else {
                        p0 *= errorRatesSamples[sampleNumber][p][j];
                        p1 *= (1 - errorRatesSamples[sampleNumber][p][j]);
                    }
                }
                labelsSamples[sampleNumber][p][i] = randomDataGenerator.nextBinomial(1, p1 / (p0 + p1));
            }
        }
    }

    private void storeSample(int sampleIndex) {
        copyArray(priorSamples[sampleIndex - 1], priorSamples[sampleIndex]);
        copyArray(labelsSamples[sampleIndex - 1], labelsSamples[sampleIndex]);
        copyArray(errorRatesSamples[sampleIndex - 1], errorRatesSamples[sampleIndex]);
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
