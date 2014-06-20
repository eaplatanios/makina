package org.platanios.learn.combination;

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
    public static boolean[][] getObservationsFromCsvFile(String filename, String separator) {
        BufferedReader br = null;
        String line;
        List<boolean[]> observations = new ArrayList<boolean[]>();

        try {
            br = new BufferedReader(new FileReader(filename));
            while ((line = br.readLine()) != null) {
                String[] outputs = line.split(separator);
                boolean[] booleanOutputs = new boolean[outputs.length];
                for (int i = 0; i < outputs.length; i++) {
                    booleanOutputs[i] = !outputs[i].equals("0");
                }
                observations.add(booleanOutputs);
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

        return observations.toArray(new boolean[observations.size()][]);
    }

    public static double[] getSampleErrorRatesFromCsvFile(String filename) {
        BufferedReader br = null;
        String line;
        List<Double> sampleErrorRates = new ArrayList<Double>();

        try {
            br = new BufferedReader(new FileReader(filename));
            while ((line = br.readLine()) != null) {
                sampleErrorRates.add(Double.parseDouble(line));
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

        Double[] sampleErrorRatesTempArray = sampleErrorRates.toArray(new Double[sampleErrorRates.size()]);
        double[] sampleErrorRatesArray = new double[sampleErrorRatesTempArray.length];

        for (int i = 0; i < sampleErrorRatesArray.length; i++) {
            sampleErrorRatesArray[i] = sampleErrorRatesTempArray[i];
        }

        return sampleErrorRatesArray;
    }
}
