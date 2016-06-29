package makina.scala.learn.evaluation

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
object BinaryClassificationEvaluation {
  private def precision(truePositives: Double, falseNegatives: Double): Double = {
    if (truePositives + falseNegatives > 0)
      truePositives / (truePositives + falseNegatives)
    else
      1.0
  }

  private def recall(truePositives: Double, falsePositivesNumber: Double): Double = {
    if (truePositives + falsePositivesNumber > 0)
      truePositives / (truePositives + falsePositivesNumber)
    else
      1.0
  }

  def areaUnderThePrecisionRecallCurve(trueLabels: Array[Boolean], predictions: Array[Double]): Double = {
    val scores = mutable.ArrayBuffer[Double]()
    var index = 0
    while (index < predictions.length) {
        var tiePredictionsCount = 0
        var tiePositivePredictionsCount = 0
        do {
            tiePredictionsCount += 1
            if (trueLabels(index))
              tiePositivePredictionsCount += 1
            index += 1
        } while (index < predictions.length && predictions(index - 1) == predictions(index))
        val previousScoresSize = scores.length
        for (i <- 0 until (index - previousScoresSize))
          scores += tiePositivePredictionsCount.toDouble / tiePredictionsCount
//        if (index < predictions.length)
//          index -= 1
    }
    var areaUnderCurve = 0.0
    var truePositives = 0.0
    var falsePositives = 0.0
    var falseNegatives = 0.0
    var previousPrecision = 0.0
    var previousRecall = 1.0
    var currentPrecision = 0.0
    var currentRecall = 0.0
    for (index <- predictions.indices)
      if (trueLabels(index))
        falseNegatives += scores(index)
    for (index <- predictions.indices) {
      val score = scores(index)
      if (trueLabels(index)) {
        falseNegatives -= score
        truePositives += score
      } else {
        falsePositives += score
      }
      currentPrecision = precision(truePositives, falseNegatives)
      currentRecall = recall(truePositives, falsePositives)
      areaUnderCurve += 0.5 * (currentPrecision - previousPrecision) * (currentRecall + previousRecall)
      previousPrecision = currentPrecision
      previousRecall = currentRecall
    }
    areaUnderCurve
  }
}
