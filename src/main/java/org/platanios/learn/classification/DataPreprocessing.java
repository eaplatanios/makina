package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataPreprocessing {
    public static TrainingData parseLabeledDataFromCSVFile(String filename) {
        String separator = ",";

        int numberOfFeatures;
        List<Integer> labels = new ArrayList<>();
        List<Vector> data = new ArrayList<>();

        BufferedReader br = null;
        String line;
        String[] classifiersNames = null;

        try {
            br = new BufferedReader(new FileReader(filename));
            line = br.readLine();
            classifiersNames = line.split(separator);
            numberOfFeatures = classifiersNames.length - 1;
            while ((line = br.readLine()) != null) {
                String[] outputs = line.split(separator);
                labels.add(Integer.parseInt(outputs[0]));
                double[] dataSample = new double[numberOfFeatures];
                for (int i = 0; i < numberOfFeatures; i++) {
                    dataSample[i] = Double.parseDouble(outputs[i+1]);
                }
                data.add(VectorFactory.buildDense(dataSample));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        if (data.size() != labels.size()) {
            throw new IllegalArgumentException("The number of data labels and data samples does not match."); // TODO: Maybe the IllegalArgumentException is not the most appropriate here.
        }

        TrainingData.Entry[] trainingData = new TrainingData.Entry[data.size()];
        for (int i = 0; i < data.size(); i++) {
            trainingData[i] = new TrainingData.Entry(data.get(i), labels.get(i));
        }

        return new TrainingData(trainingData);
    }

    public static DataInstance<Vector, Integer>[] parseBinaryLabeledDataFromCSVFile(String filename) {
        String separator = ",";

        int numberOfFeatures;
        List<DataInstance<Vector, Integer>> data = new ArrayList<>();

        BufferedReader br = null;
        String line;

        try {
            br = new BufferedReader(new FileReader(filename));
            line = br.readLine();
            numberOfFeatures = line.split(separator).length - 1;
            while ((line = br.readLine()) != null) {
                String[] outputs = line.split(separator);
                SparseVector features = (SparseVector) VectorFactory.build(numberOfFeatures, VectorType.SPARSE);
                int label = Integer.parseInt(outputs[0]);
                for (int i = 0; i < numberOfFeatures; i++) {
                    features.set(i, Double.parseDouble(outputs[i + 1]));
                }
                data.add(new DataInstance<Vector, Integer>(features, label));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return data.toArray(new DataInstance[data.size()]);
    }

    public static DataInstance<Vector, Integer>[] parseLabeledDataFromLIBSVMFile(String filename,
                                                                                 boolean sparseFeatures) {
        String separator = " ";

        List<DataInstance<Vector, Integer>> data = new ArrayList<>();

        BufferedReader br = null;
        String line;

        try {
            br = new BufferedReader(new FileReader(filename));
            while ((line = br.readLine()) != null) {
                String[] tokens = line.split(separator);
                int label = Integer.parseInt(tokens[0]) - 1;
                Vector features;
                if (sparseFeatures) {
                    features = VectorFactory.build(54, VectorType.SPARSE);
                } else {
                    features = VectorFactory.build(54, VectorType.DENSE);
                }
                for (int i = 1; i < tokens.length; i++) {
                    String[] featurePair = tokens[i].split(":");
                    features.set(Integer.parseInt(featurePair[0]) - 1, Double.parseDouble(featurePair[1]));
                }
                data.add(new DataInstance<Vector, Integer>(features, label));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return data.toArray(new DataInstance[data.size()]);
    }
}
