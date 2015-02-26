package org.platanios.experiment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationMissingData {
    public static void main(String[] args) {

    }

    public static DomainData parseLabeledDataFromCSVFile(
            File file,
            String separator,
            double[] classificationThresholds,
            int subSampling
    ) {
        BufferedReader br = null;
        String line;
        List<boolean[]> classifiersOutputsList = new ArrayList<>();
        List<Boolean> trueLabelsList = new ArrayList<>();

        try {
            br = new BufferedReader(new FileReader(file));
            br.readLine();
            int numberOfSamplesRead = 0;
            while ((line = br.readLine()) != null) {
                if (numberOfSamplesRead % subSampling == 0) {
                    String[] outputs = line.split(separator);
                    trueLabelsList.add(!outputs[0].equals("0"));
                    boolean[] booleanOutputs = new boolean[outputs.length - 1];
                    for (int i = 1; i < outputs.length; i++) {
                        if (classificationThresholds == null) {
                            booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= 0.5;
                        } else if (classificationThresholds.length == 1) {
                            booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= classificationThresholds[0];
                        } else {
                            booleanOutputs[i - 1] = Double.parseDouble(outputs[i]) >= classificationThresholds[i - 1];
                        }
                    }
                    classifiersOutputsList.add(booleanOutputs);
                }
                numberOfSamplesRead++;
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

        boolean[] trueLabels = new boolean[trueLabelsList.size()];
        for (int i = 0; i < trueLabels.length; i++)
            trueLabels[i] = trueLabelsList.get(i);

        return new DomainData(classifiersOutputsList.toArray(new boolean[classifiersOutputsList.size()][]), trueLabels);
    }

    private static class DomainData {
        private boolean[][] functionOutputs;
        private boolean[] trueLabels;

        protected DomainData(boolean[][] functionOutputs, boolean[] trueLabels) {
            this.functionOutputs = functionOutputs;
            this.trueLabels = trueLabels;
        }

        protected boolean[][] getFunctionOutputs() {
            return functionOutputs;
        }

        protected boolean[] getTrueLabels() {
            return trueLabels;
        }
    }
}
