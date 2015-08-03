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
public class DiscountedCumulativeGain<T extends Vector, S> extends CurveEvaluation<T, S> {
    private final boolean useAlternativeFormulation;

    public DiscountedCumulativeGain(boolean useAlternativeFormulation) {
        super();
        this.useAlternativeFormulation = useAlternativeFormulation;
    }

    @Override
    public void addResult(String name,
                          List<PredictedDataInstance<T, S>> predictions,
                          Function<PredictedDataInstance<T, S>, Boolean> groundTruth) {
        Collections.sort(predictions, Comparator.comparing(PredictedDataInstance::probability));
        List<CurvePoint> points = new ArrayList<>();
        double areaUnderCurve = 0;
        for (int predictionIndex = 0; predictionIndex < predictions.size(); predictionIndex++) {
            double relevantPrediction = groundTruth.apply(predictions.get(predictionIndex)) ? 1 : 0;
            if (!useAlternativeFormulation) {
                if (points.size() == 0) {
                    points.add(new CurvePoint(predictionIndex + 1, relevantPrediction));
                    continue;
                }
                points.add(new CurvePoint(predictionIndex + 1,
                                          points.get(points.size() - 1).getVerticalAxisValue()
                                                  + relevantPrediction * Math.log(2) / Math.log(predictionIndex + 1)));
            } else {
                if (points.size() == 0) {
                    points.add(new CurvePoint(predictionIndex + 1,
                                              (Math.pow(2, relevantPrediction) - 1)
                                                      * Math.log(2)
                                                      / Math.log(predictionIndex + 2)));
                    continue;
                }
                points.add(new CurvePoint(predictionIndex + 1,
                                          points.get(points.size() - 1).getVerticalAxisValue()
                                                  + (Math.pow(2, relevantPrediction) - 1) * Math.log(2)
                                                  / Math.log(predictionIndex + 2)));
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
        return "Discounted Cumulative Gain Curve";
    }

    @Override
    protected String getHorizontalAxisName() {
        return "Rank Position";
    }

    @Override
    protected String getVerticalAxisName() {
        return "Discounted Cumulative Gain";
    }
}
