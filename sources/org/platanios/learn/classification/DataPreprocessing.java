package org.platanios.learn.classification;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
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
        List<RealVector> data = new ArrayList<>();

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
                data.add(new ArrayRealVector(dataSample));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
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

        return new TrainingData(data.toArray(new RealVector[data.size()]), labels.toArray(new Integer[labels.size()]));
    }
}
