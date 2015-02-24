package org.platanios.experiment;

import org.platanios.learn.classification.reflection.ErrorEstimationDomainsDPGraphicalModelComplicated;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ErrorEstimationDomainsDPGraphicalModelDebugging {
    public static void main(String[] args) {
        int numberOfSamples = 1000;
        int numberOfFunctions = 3;
        List<boolean[][]> functionOutputs = new ArrayList<>();
        boolean[][] domain1FunctionOutputs = new boolean[numberOfSamples][numberOfFunctions];
        boolean[][] domain2FunctionOutputs = new boolean[numberOfSamples][numberOfFunctions];
        boolean[][] domain3FunctionOutputs = new boolean[numberOfSamples][numberOfFunctions];
        boolean[][] domain4FunctionOutputs = new boolean[numberOfSamples][numberOfFunctions];
        for (int i = 0; i < numberOfSamples; i++) {
            if (i < numberOfSamples / 2) {
                for (int j = 0; j < numberOfFunctions - 1; j++) {
                    domain1FunctionOutputs[i][j] = true;
                    domain2FunctionOutputs[i][j] = false;
                    domain3FunctionOutputs[i][j] = false;
                    domain4FunctionOutputs[i][j] = true;
                }
                domain1FunctionOutputs[i][2] = true;
                domain2FunctionOutputs[i][2] = true;
                domain3FunctionOutputs[i][2] = false;
                domain4FunctionOutputs[i][2] = true;
            } else {
                for (int j = 0; j < numberOfFunctions - 1; j++) {
                    domain1FunctionOutputs[i][j] = false;
                    domain2FunctionOutputs[i][j] = true;
                    domain3FunctionOutputs[i][j] = false;
                    domain4FunctionOutputs[i][j] = false;
                }
                domain1FunctionOutputs[i][2] = false;
                domain2FunctionOutputs[i][2] = false;
                domain3FunctionOutputs[i][2] = false;
                domain4FunctionOutputs[i][2] = false;
            }
        }
        functionOutputs.add(domain1FunctionOutputs);
        functionOutputs.add(domain2FunctionOutputs);
        functionOutputs.add(domain3FunctionOutputs);
        functionOutputs.add(domain4FunctionOutputs);
        ErrorEstimationDomainsDPGraphicalModelComplicated eeddpgm = new ErrorEstimationDomainsDPGraphicalModelComplicated(functionOutputs, 1000, null);
        eeddpgm.performGibbsSampling();
    }
}
