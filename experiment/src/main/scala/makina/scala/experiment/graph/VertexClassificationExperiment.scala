package makina.scala.experiment.graph

import breeze.linalg.DenseVector
import breeze.stats
import com.typesafe.scalalogging.Logger
import makina.scala.experiment.graph.VertexClassificationExperiment._
import makina.scala.learn.graph.{Graph, Vertex}
import makina.scala.learn.neural.graph.DeepGraph.{CrossEntropyLossFunction, UpdateFunction, VertexClassificationOutputFunction, VertexContent}
import makina.scala.learn.neural.graph.UpdateFunction
import DataSets.LabeledSet
import makina.scala.learn.evaluation.BinaryClassificationEvaluation.areaUnderThePrecisionRecallCurve
import makina.scala.learn.neural.graph.{DeepGraph, UpdateFunction}
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.util.Random

/**
  * @author Emmanouil Antonios Platanios
  */
object VertexClassificationExperiment {
  val logger = Logger(LoggerFactory.getLogger("Experiment / Deep Graph Vertex Classification"))

  def apply(dataSet: LabeledSet, updateFunctionName: String, resultsFolder: String) =
    new VertexClassificationExperiment(dataSet, updateFunctionName, resultsFolder)

  trait Results {
    def average(results: Array[Results]): Results
    override def toString: String
  }

  object BinaryResults extends Results {
    override def average(results: Array[Results]): Results = {
      BinaryResults(results.map(_.asInstanceOf[BinaryResults].trainAUC).toList.mean,
                    results.map(_.asInstanceOf[BinaryResults].testAUC).toList.mean,
                    results.map(_.asInstanceOf[BinaryResults].overallAUC).toList.mean,
                    results.map(_.asInstanceOf[BinaryResults].trainAUC).toList.stddev,
                    results.map(_.asInstanceOf[BinaryResults].testAUC).toList.stddev,
                    results.map(_.asInstanceOf[BinaryResults].overallAUC).toList.stddev)
    }
  }

  case class BinaryResults(trainAUC: Double,
                           testAUC: Double,
                           overallAUC: Double,
                           trainAUCStd: Double = 0.0,
                           testAUCStd: Double = 0.0,
                           overallAUCStd: Double = 0.0) extends Results {
    override def average(results: Array[Results]): Results = BinaryResults.average(results)
    override def toString: String =
      f"$trainAUC \u00B1 $trainAUCStd | $testAUC \u00B1 $testAUCStd | $overallAUC \u00B1 $overallAUCStd"
  }

  implicit class ImplDoubleVecUtils(values: Seq[Double]) {
    def mean   = stats.mean(DenseVector[Double](values.toArray))
    def stddev = stats.stddev(DenseVector[Double](values.toArray))
  }

  private def appendResultsToFile(featuresSize: Int, numberOfSteps: Int, results: Results, resultsFolder: String) = {}

  def main(args: Array[String]) = {
    val dataSet    = DataSets.loadLabeledSet(args(5), args(6).equals("1"))
    val experiment = VertexClassificationExperiment(dataSet, args(0), args(7))
    experiment.run(args(1).split(",").map(_.toInt), args(2).split(",").map(_.toInt), args(3).toInt, args(4).toInt)
  }
}

class VertexClassificationExperiment private (private val dataSet: LabeledSet,
                                              private val updateFunctionName: String,
                                              private val resultsFolder: String) {
  logger.info("Number of vertices: " + dataSet.vertices.size)
  logger.info("Number of edges: " + dataSet.edges.size)
  val graph = Graph[VertexContent, Null]()
  val vertexIdsMap = mutable.Map[Int, Vertex[VertexContent, Null]]()
  dataSet.vertices.foreach(v => vertexIdsMap += v -> Vertex[VertexContent, Null](VertexContent(v, 0)))
  dataSet.edges.foreach(e => graph.addEdge(vertexIdsMap.get(e.sourceVertex).get, vertexIdsMap.get(e.destinationVertex).get))
  logger.info("Finished generating graphs for the " + dataSet.name + " data set.")
  val trueLabels: Map[Int, Int] = dataSet.vertexLabels.toMap

  def run(featuresSize: Array[Int], numberOfSteps: Array[Int], numberOfFolds: Int, numberOfFoldsToKeep: Int) = {
    val numberOfLabels = trueLabels.values.toSet.size
    val results        = Array.ofDim[Results](featuresSize.length, numberOfSteps.length)
    for (i <- featuresSize.indices; j <- numberOfSteps.indices) {
      val currentResults                       = Array.ofDim[Results](numberOfFoldsToKeep)
      val shuffledTrueLabels: List[(Int, Int)] = Random.shuffle(trueLabels.toList).toList
      val foldSize                             = Math.floorDiv(shuffledTrueLabels.length, numberOfFolds)
      for (fold <- 0 until numberOfFoldsToKeep) {
        val trainingData = mutable.Map[Int, DenseVector[Double]]()
        shuffledTrueLabels
          .slice(0, fold * foldSize)
          .foreach(l => trainingData += l._1 -> DenseVector[Double](l._2.toDouble))
        shuffledTrueLabels
          .slice((fold + 1) * foldSize, shuffledTrueLabels.length)
          .foreach(l => trainingData += l._1 -> DenseVector[Double](l._2.toDouble))
        if (numberOfLabels > 2) {
          throw new UnsupportedOperationException("Multi-class classification experiments not supported yet.")
        } else {
          currentResults(fold) = runBinaryExperiment(featuresSize(i), numberOfSteps(j), trainingData.toMap)
        }
      }
      results(i)(j) = BinaryResults.average(currentResults)
      appendResultsToFile(featuresSize(i), numberOfSteps(j), results(i)(j), resultsFolder)
    }
    logger.info(f"Results for the ${dataSet.name} data set:")
    for (i <- featuresSize.indices; j <- numberOfSteps.indices) {
      val binaryResults = results(i)(j).asInstanceOf[BinaryResults]
      logger.info(f"\tF = ${featuresSize(i)}, K = ${numberOfSteps(j)}:\t { " +
                  f"Train AUC: ${binaryResults.trainAUC}%20s | " +
                  f"Test AUC: ${binaryResults.testAUC}%20s | " +
                  f"Overall AUC: ${binaryResults.overallAUC}%20s }")
    }
  }

  def runBinaryExperiment(featuresSize: Int,
                          numberOfSteps: Int,
                          trainingData: Map[Int, DenseVector[Double]]): Results = {
    val algorithm = DeepGraph[Null](
        featuresSize,
        1,
        numberOfSteps,
        graph,
        UpdateFunction[Null](updateFunctionName, featuresSize, graph.vertices.toList),
        VertexClassificationOutputFunction(featuresSize, 1),
        CrossEntropyLossFunction(1)
    )
    algorithm.reset()
    logger.info(s"Training for the ${dataSet.name} data set.")
    algorithm.train(trainingData)
    logger.info(s"Finished training for the ${dataSet.name} data set.")
    logger.info(s"Evaluating for the ${dataSet.name} data set.")
    algorithm.forwardPass()
    val trainPairs = mutable.ArrayBuffer[(Boolean, Double)]()
    val testPairs  = mutable.ArrayBuffer[(Boolean, Double)]()
    graph.vertices
      .filter(v => trueLabels.contains(v.content.id))
      .seq
      .foreach(v => {
          val pair = (if (trueLabels.get(v.content.id).get == 1) true else false, algorithm.output(v)(0))
          if (trainingData.contains(v.content.id)) trainPairs += pair
          else testPairs += pair
      })
    evaluateBinary(trainPairs.toList, testPairs.toList)
  }

  private def evaluateBinary(trainPairs: List[(Boolean, Double)], testPairs: List[(Boolean, Double)]): Results = {
    val sortedTrainPairs   = trainPairs.sortBy(-_._2)
    val sortedTestPairs    = testPairs.sortBy(-_._2)
    val sortedOverallPairs = (trainPairs ++ testPairs).sortBy(-_._2)
    BinaryResults(
        areaUnderThePrecisionRecallCurve(sortedTrainPairs.map(_._1).toArray, sortedTrainPairs.map(_._2).toArray),
        areaUnderThePrecisionRecallCurve(sortedTestPairs.map(_._1).toArray, sortedTestPairs.map(_._2).toArray),
        areaUnderThePrecisionRecallCurve(sortedOverallPairs.map(_._1).toArray, sortedOverallPairs.map(_._2).toArray))
  }
}
