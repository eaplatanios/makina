package org.platanios.experiment;

import org.platanios.learn.data.MultiViewDataSet;
import org.platanios.learn.data.MultiViewPredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Integrator {
    private final String featureMapsDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/features";
    private final String trainingDataFilePath = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/varias_data/trainData/cpl/fruit.txt";

    private MultiViewDataSet<MultiViewPredictedDataInstance<Vector, Integer>> labeledData;
    private MultiViewDataSet<MultiViewPredictedDataInstance<Vector, Integer>> trainingData;

    private void loadFeatureMaps() {

    }

    private void loadLabeledNPsData() {

    }

    public static void main(String[] args) {

    }
}
