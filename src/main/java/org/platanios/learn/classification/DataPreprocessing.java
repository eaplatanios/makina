package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorFactory;
import org.platanios.learn.math.matrix.VectorType;

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
                data.add(VectorFactory.build(dataSample, VectorType.DENSE));
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
}
