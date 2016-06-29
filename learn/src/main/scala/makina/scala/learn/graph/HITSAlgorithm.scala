package makina.scala.learn.graph

import com.typesafe.scalalogging.Logger
import makina.scala.learn.graph.HITSAlgorithm._
import org.slf4j.LoggerFactory

/**
  * @author Emmanouil Antonios Platanios
  */
case class HITSAlgorithm[E >: Null <: AnyRef](
  graph: Graph[VertexContent, E],
  maximumNumberOfIterations: Int = 30,
  checkForScoresConvergence: Boolean = true,
  scoresChangeTolerance: Double = 1e-6,
  loggingLevel: Int = 0
) {
  private var currentIteration: Int = 0
  private var lastMeanAbsoluteScoresChange: Double = Double.MaxValue
  private var authorityNormalizationConstant: Double = 0.0
  private var hubNormalizationConstant: Double = 0.0

  def computeRanks() = {
    while (!checkTerminationConditions()) {
      performIterationUpdates()
      currentIteration += 1
      if ((loggingLevel == 1 && currentIteration % 1000 == 0) ||
          (loggingLevel == 2 && currentIteration % 100 == 0) ||
          (loggingLevel == 3 && currentIteration % 10 == 0) ||
          loggingLevel > 3)
        logIteration()
    }
  }

  private def checkTerminationConditions(): Boolean = {
    currentIteration >= maximumNumberOfIterations ||
    (checkForScoresConvergence && lastMeanAbsoluteScoresChange <= scoresChangeTolerance)
  }

  // TODO: Can be made more efficient by calling the update vertices content method only once.
  private def performIterationUpdates() = {
    graph.computeVerticesUpdatedContent(vertexUpdateFunction)
    graph.updateVerticesContent()
    authorityNormalizationConstant = Math.sqrt(authorityNormalizationConstant)
    hubNormalizationConstant = Math.sqrt(hubNormalizationConstant)
    graph.computeVerticesUpdatedContent(normalizationVertexUpdateFunction)
    graph.updateVerticesContent()
    authorityNormalizationConstant = 0.0
    hubNormalizationConstant = 0.0
    if (checkForScoresConvergence) {
      lastMeanAbsoluteScoresChange = 0.0
      for (vertex <- graph.vertices)
        lastMeanAbsoluteScoresChange += vertex.content.lastAbsoluteChange
      lastMeanAbsoluteScoresChange /= graph.numberOfVertices
    }
  }

  private def logIteration() = {
    if (checkForScoresConvergence)
      logger.info(f"Iteration #: $currentIteration%10d | " +
                  f"Mean Absolute Scores Change: $lastMeanAbsoluteScoresChange%20s")
    else
      logger.info(f"Iteration #: $currentIteration%10d")
  }

  private def vertexUpdateFunction(vertex: Vertex[VertexContent, E]): VertexContent = {
    val oldAuthorityScore: Double = vertex.content.authorityScore
    val oldHubScore: Double = vertex.content.hubScore
    var newAuthorityScore: Double = 0.0
    var newHubScore: Double = 0.0
    vertex.incomingEdges.foreach(newAuthorityScore += _.source.content.hubScore)
    vertex.outgoingEdges.foreach(newHubScore += _.destination.content.authorityScore)
    authorityNormalizationConstant += newAuthorityScore * newAuthorityScore
    hubNormalizationConstant += newHubScore * newHubScore
    VertexContent(
      vertex.content.id,
      newAuthorityScore,
      newHubScore,
      if (checkForScoresConvergence)
        Math.abs(newAuthorityScore - oldAuthorityScore) + Math.abs(newHubScore - oldHubScore)
      else
        0.0
    )
  }

  private def normalizationVertexUpdateFunction(vertex: Vertex[VertexContent, E]): VertexContent = {
    VertexContent(
      vertex.content.id,
      vertex.content.authorityScore / authorityNormalizationConstant,
      vertex.content.hubScore / hubNormalizationConstant,
      vertex.content.lastAbsoluteChange
    )
  }
}

object HITSAlgorithm {
  val logger = Logger(LoggerFactory.getLogger("Graph / HITS"))

  case class VertexContent(id: Int,
                           authorityScore: Double,
                           hubScore: Double,
                           lastAbsoluteChange: Double = Double.MaxValue)
}
