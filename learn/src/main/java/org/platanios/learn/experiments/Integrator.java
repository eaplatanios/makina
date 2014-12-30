package org.platanios.learn.experiments;

import org.platanios.learn.data.MultiViewPredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Integrator {
    private final String featureMapsDirectory = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/features";
    private final String trainingDataFilePath = "/Volumes/Macintosh HD/Users/Anthony/Development/NELL/data/varias_data/trainData/cpl/fruit.txt";

    private List<MultiViewPredictedDataInstance<Vector, Integer>> labeledData;
    private List<MultiViewPredictedDataInstance<Vector, Integer>> trainingData;

    private void loadFeatureMaps() {

    }

    private void loadLabeledNPsData() {
        labeledData = new ArrayList<>();
    }

    public static void main(String[] args) {

    }
}
