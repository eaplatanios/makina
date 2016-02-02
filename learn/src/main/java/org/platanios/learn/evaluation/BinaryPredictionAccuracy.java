package org.platanios.learn.evaluation;

import org.platanios.learn.data.DataSet;
import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.MathUtilities;
import org.platanios.learn.math.matrix.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class BinaryPredictionAccuracy<T extends Vector, S> {
    protected static final double epsilon = MathUtilities.computeMachineEpsilonDouble();

    protected final Map<String, Double> predictionAccuracies = new HashMap<>();

    protected final double classificationThreshold;

    public BinaryPredictionAccuracy() {
        this(0.5);
    }

    public BinaryPredictionAccuracy(double classificationThreshold) {
        this.classificationThreshold = classificationThreshold;
    }

    // TODO: It is maybe better to have a getList() method in the data set classes.
    // TODO: Assumes that the labels of all predicted data instances are positive.
    public void addResult(String name,
                          DataSet<PredictedDataInstance<T, S>> predictions,
                          Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        List<PredictedDataInstance<T, S>> predictionsList = new ArrayList<>();
        predictions.forEach(predictionsList::add);
        addResult(name, predictionsList, groundTruth);
    }

    public void addResult(String name,
                          List<PredictedDataInstance<T, S>> predictions,
                          Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        double predictionAccuracy = 0;
        for (PredictedDataInstance<T, S> prediction : predictions)
            if (prediction.probability() > classificationThreshold && groundTruth.apply(prediction)
                    || prediction.probability() <= classificationThreshold && !groundTruth.apply(prediction))
                predictionAccuracy++;
        predictionAccuracies.put(name, predictionAccuracy / predictions.size());
    }

    public Map<String, Double> getPredictionAccuracies() {
        return predictionAccuracies;
    }

    public double getPredictionAccuracy(String name) {
        return predictionAccuracies.get(name);
    }
}
