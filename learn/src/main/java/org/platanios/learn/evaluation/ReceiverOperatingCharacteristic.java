package org.platanios.learn.evaluation;

import org.platanios.learn.data.PredictedDataInstance;
import org.platanios.learn.math.matrix.Vector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

/**
 * @author Emmanouil Antonios Platanios
 */
public class ReceiverOperatingCharacteristic<T extends Vector, S> extends CurveEvaluation<T, S> {
    private final int numberOfCurvePoints;

    public ReceiverOperatingCharacteristic() {
        this(-1);
    }

    public ReceiverOperatingCharacteristic(int numberOfCurvePoints) {
        super();
        this.numberOfCurvePoints = numberOfCurvePoints;
    }

    @Override
    public void addResult(String name,
                          List<PredictedDataInstance<T, S>> predictions,
                          Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        List<CurvePoint> points = new ArrayList<>();
        Collections.sort(predictions, Comparator.comparing(PredictedDataInstance::probability));
        int truePositivesNumber = 0;
        int trueNegativesNumber = 0;
        int falsePositivesNumber = 0;
        int falseNegativesNumber = 0;
        for (PredictedDataInstance<T, S> prediction : predictions)
            if (groundTruth.apply(prediction))
                falseNegativesNumber++;
            else
                trueNegativesNumber++;
        points.add(new CurvePoint(
                (falsePositivesNumber + epsilon) / (falsePositivesNumber + trueNegativesNumber + epsilon),
                (truePositivesNumber + epsilon) / (truePositivesNumber + falseNegativesNumber + epsilon)
        ));
        double areaUnderCurve = 0;
        if (numberOfCurvePoints < 0) { // TODO: This is not totally correct -- think of what happens when two predictions have the same probability.
            for (PredictedDataInstance<T, S> prediction : predictions) {
                if (groundTruth.apply(prediction)) {
                    truePositivesNumber++;
                    falseNegativesNumber--;
                } else {
                    falsePositivesNumber++;
                    trueNegativesNumber--;
                }
                points.add(new CurvePoint(
                        (falsePositivesNumber + epsilon) / (falsePositivesNumber + trueNegativesNumber + epsilon),
                        (truePositivesNumber + epsilon) / (truePositivesNumber + falseNegativesNumber + epsilon)
                ));
                int k = points.size() - 1;
                areaUnderCurve += 0.5
                        * (points.get(k).getHorizontalAxisValue() - points.get(k - 1).getHorizontalAxisValue())
                        * (points.get(k).getVerticalAxisValue() + points.get(k - 1).getVerticalAxisValue());
            }
        } else {
            int previousThresholdPredictionIndex = 0;
            for (double thresholdIndex = 1; thresholdIndex < numberOfCurvePoints; thresholdIndex++) {
                double threshold = 1 - thresholdIndex / (numberOfCurvePoints - 1);
                PredictedDataInstance<T, S> prediction = predictions.get(previousThresholdPredictionIndex);
                while (prediction.probability() >= threshold) {
                    if (groundTruth.apply(prediction)) {
                        truePositivesNumber++;
                        falseNegativesNumber--;
                    } else {
                        falsePositivesNumber++;
                        trueNegativesNumber--;
                    }
                    if (++previousThresholdPredictionIndex < predictions.size())
                        prediction = predictions.get(previousThresholdPredictionIndex);
                    else
                        break;
                }
                points.add(new CurvePoint(
                        (falsePositivesNumber + epsilon) / (falsePositivesNumber + trueNegativesNumber + epsilon),
                        (truePositivesNumber + epsilon) / (truePositivesNumber + falseNegativesNumber + epsilon)
                ));
                int k = points.size() - 1;
                areaUnderCurve += 0.5
                        * (points.get(k).getHorizontalAxisValue() - points.get(k - 1).getHorizontalAxisValue())
                        * (points.get(k).getVerticalAxisValue() + points.get(k - 1).getVerticalAxisValue());
                if (previousThresholdPredictionIndex == predictions.size())
                    break;
            }
        }
        curves.add(new Curve(name, points));
        areaUnderCurves.add(areaUnderCurve);
    }

    @Override
    protected String getPlotTitle() {
        return "Receiver Operating Characteristic Curve";
    }

    @Override
    protected String getHorizontalAxisName() {
        return "False Positive Rate";
    }

    @Override
    protected String getVerticalAxisName() {
        return "True Positive Rate";
    }
}
