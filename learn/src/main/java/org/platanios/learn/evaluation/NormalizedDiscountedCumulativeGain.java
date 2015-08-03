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
public class NormalizedDiscountedCumulativeGain<T extends Vector, S> extends CurveEvaluation<T, S> {
    private final boolean useAlternativeFormulation;

    public NormalizedDiscountedCumulativeGain(boolean useAlternativeFormulation) {
        super();
        this.useAlternativeFormulation = useAlternativeFormulation;
    }

    @Override
    public void addResult(String name,
                          List<PredictedDataInstance<T, S>> predictions,
                          Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        int totalNumberOfPositiveLabels = 0;
        for (PredictedDataInstance<T, S> prediction : predictions)
            if (groundTruth.apply(prediction))
                totalNumberOfPositiveLabels++;
        Collections.sort(predictions, Comparator.comparing(PredictedDataInstance::probability));
        List<CurvePoint> points = new ArrayList<>();
        double areaUnderCurve = 0;
        double idealDiscountedCumulativeGain = 0;
        for (int predictionIndex = 0; predictionIndex < predictions.size(); predictionIndex++) {
            double relevantPrediction = groundTruth.apply(predictions.get(predictionIndex)) ? 1 : 0;
            if (!useAlternativeFormulation) {
                if (points.size() == 0) {
                    if (predictionIndex < totalNumberOfPositiveLabels)
                        idealDiscountedCumulativeGain++;
                    points.add(new CurvePoint(predictionIndex + 1, relevantPrediction / idealDiscountedCumulativeGain));
                    continue;
                }
                double discountedCumulativeGain =
                        points.get(points.size() - 1).getVerticalAxisValue() * idealDiscountedCumulativeGain
                        + relevantPrediction * Math.log(2) / Math.log(predictionIndex + 1);
                if (predictionIndex < totalNumberOfPositiveLabels)
                    idealDiscountedCumulativeGain += Math.log(2) / Math.log(predictionIndex + 1);
                points.add(new CurvePoint(predictionIndex + 1,
                                          discountedCumulativeGain / idealDiscountedCumulativeGain));
            } else {
                if (points.size() == 0) {
                    if (predictionIndex < totalNumberOfPositiveLabels)
                        idealDiscountedCumulativeGain += Math.log(2) / Math.log(predictionIndex + 2);
                    double discountedCumulativeGain =
                            (Math.pow(2, relevantPrediction) - 1) * Math.log(2) / Math.log(predictionIndex + 2);
                    points.add(new CurvePoint(predictionIndex + 1,
                                              discountedCumulativeGain / idealDiscountedCumulativeGain));
                    continue;
                }
                double discountedCumulativeGain =
                        points.get(points.size() - 1).getVerticalAxisValue() * idealDiscountedCumulativeGain
                        + (Math.pow(2, relevantPrediction) - 1) * Math.log(2) / Math.log(predictionIndex + 2);
                if (predictionIndex < totalNumberOfPositiveLabels)
                    idealDiscountedCumulativeGain += Math.log(2) / Math.log(predictionIndex + 2);
                points.add(new CurvePoint(predictionIndex + 1,
                                          discountedCumulativeGain / idealDiscountedCumulativeGain));
            }
            int k = points.size() - 1;
            areaUnderCurve += 0.5
                    * (points.get(k).getHorizontalAxisValue() - points.get(k - 1).getHorizontalAxisValue())
                    * (points.get(k).getVerticalAxisValue() + points.get(k - 1).getVerticalAxisValue());
        }
        curves.add(new Curve(name, points));
        areaUnderCurves.add(areaUnderCurve);
    }

    @Override
    protected String getPlotTitle() {
        return "Normalized Discounted Cumulative Gain Curve";
    }

    @Override
    protected String getHorizontalAxisName() {
        return "Rank Position";
    }

    @Override
    protected String getVerticalAxisName() {
        return "Normalized Discounted Cumulative Gain";
    }
}
