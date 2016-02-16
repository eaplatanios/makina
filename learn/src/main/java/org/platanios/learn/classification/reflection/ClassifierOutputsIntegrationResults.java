package org.platanios.learn.classification.reflection;

import org.platanios.learn.classification.Label;
import org.platanios.learn.data.DataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ClassifierOutputsIntegrationResults {
    private final Map<DataInstance<Vector>, Map<Label, Double>> integratedDataSet;
    private final Map<Label, Map<Integer, Double>> errorRates;

    public ClassifierOutputsIntegrationResults(Map<DataInstance<Vector>, Map<Label, Double>> integratedDataSet,
                                               Map<Label, Map<Integer, Double>> errorRates) {
        this.integratedDataSet = integratedDataSet;
        this.errorRates = errorRates;
    }

    public Map<DataInstance<Vector>, Map<Label, Double>> getIntegratedDataSet() {
        return integratedDataSet;
    }

    public Map<Label, Map<Integer, Double>> getErrorRates() {
        return errorRates;
    }
}
