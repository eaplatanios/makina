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
public class PrecisionRecall<T extends Vector, S> extends CurveEvaluation<T, S> {
    private final int numberOfCurvePoints;

    public PrecisionRecall() {
        this(-1);
    }

    public PrecisionRecall(int numberOfCurvePoints) {
        super();
        this.numberOfCurvePoints = numberOfCurvePoints;
    }

    @Override
    public void addResult(String name,
                          List<PredictedDataInstance<T, S>> predictions,
                          Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        if (predictions.size() == 0) { // TODO: Do that for other evaluation curves as well.
            List<CurvePoint> points = new ArrayList<>();
            points.add(new CurvePoint(0, 1));
            points.add(new CurvePoint(1, 1));
            curves.put(name, new Curve(name, points));
            areaUnderCurves.put(name, 1.0);
            return;
        }
        Collections.sort(predictions,
                         Collections.reverseOrder(Comparator.comparing(PredictedDataInstance::probability)));
        List<CurvePoint> points = new ArrayList<>();
        int truePositivesNumber = 0;
        int falsePositivesNumber = 0;
        int falseNegativesNumber = 0;
        for (PredictedDataInstance<T, S> prediction : predictions)
            if (groundTruth.apply(prediction))
                falseNegativesNumber++;
        points.add(new CurvePoint(
                computePrecision(truePositivesNumber, falseNegativesNumber),
                computeRecall(truePositivesNumber, falsePositivesNumber)
        ));
        double areaUnderCurve = 0;
        if (numberOfCurvePoints < 0) { // TODO: Maybe fixed!!! This is not totally correct -- think of what happens when two predictions have the same probability.
            for (int predictionIndex = 0; predictionIndex < predictions.size(); predictionIndex++) {
                do {
                    if (groundTruth.apply(predictions.get(predictionIndex))) {
                        falseNegativesNumber--;
                        truePositivesNumber++;
                    } else {
                        falsePositivesNumber++;
                    }
                    predictionIndex++;
                    if (predictionIndex == predictions.size())
                        break;
                } while (predictions.get(predictionIndex - 1).probability()
                        == predictions.get(predictionIndex).probability());
                if (predictionIndex < predictions.size())
                    predictionIndex--;
                points.add(new CurvePoint(
                        computePrecision(truePositivesNumber, falseNegativesNumber),
                        computeRecall(truePositivesNumber, falsePositivesNumber)
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
                        falseNegativesNumber--;
                        truePositivesNumber++;
                    } else {
                        falsePositivesNumber++;
                    }
                    if (++previousThresholdPredictionIndex < predictions.size())
                        prediction = predictions.get(previousThresholdPredictionIndex);
                    else
                        break;
                }
                points.add(new CurvePoint(
                        computePrecision(truePositivesNumber, falseNegativesNumber),
                        computeRecall(truePositivesNumber, falsePositivesNumber)
                ));
                int k = points.size() - 1;
                areaUnderCurve += 0.5
                        * (points.get(k).getHorizontalAxisValue() - points.get(k - 1).getHorizontalAxisValue())
                        * (points.get(k).getVerticalAxisValue() + points.get(k - 1).getVerticalAxisValue());
                if (previousThresholdPredictionIndex == predictions.size())
                    break;
            }
        }
        curves.put(name, new Curve(name, points));
        areaUnderCurves.put(name, areaUnderCurve);
    }

    private double computePrecision(int truePositivesNumber, int falseNegativesNumber) {
        if (truePositivesNumber + falseNegativesNumber > 0)
            return (double) truePositivesNumber / (truePositivesNumber + falseNegativesNumber);
        else
            return 1.0;
    }

    private double computeRecall(int truePositivesNumber, int falsePositivesNumber) {
        if (truePositivesNumber + falsePositivesNumber > 0)
            return (double) truePositivesNumber / (truePositivesNumber + falsePositivesNumber);
        else
            return 1.0;
    }

    @Override
    protected String getPlotTitle() {
        return "Precision-Recall Curve";
    }

    @Override
    protected String getHorizontalAxisName() {
        return "Recall";
    }

    @Override
    protected String getVerticalAxisName() {
        return "Precision";
    }
}
