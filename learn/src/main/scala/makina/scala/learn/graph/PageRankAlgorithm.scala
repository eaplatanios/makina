package makina.scala.learn.graph

import com.typesafe.scalalogging.Logger
import makina.scala.learn.graph.PageRankAlgorithm._
import org.slf4j.LoggerFactory

/**
  * @author Emmanouil Antonios Platanios
  */
case class PageRankAlgorithm[E >: Null <: AnyRef](
  graph: Graph[VertexContent, E],
  dampingFactor: Double = 0.85,
  maximumNumberOfIterations: Int = 30,
  checkForRankConvergence: Boolean = true,
  rankChangeTolerance: Double = 1e-6,
  loggingLevel: Int = 0
) {
  val dampingTerm = (1.0 - dampingFactor) / graph.numberOfVertices

  private var currentIteration: Int              = 0
  private var lastMeanAbsoluteRankChange: Double = Double.MaxValue

  def computeRanks() = {
    while (!checkTerminationConditions()) {
      performIterationUpdates()
      currentIteration += 1
      if ((loggingLevel == 1 && currentIteration % 1000 == 0) || (loggingLevel == 2 && currentIteration % 100 == 0) ||
          (loggingLevel == 3 && currentIteration % 10 == 0) || loggingLevel > 3) logIteration()
    }
  }

  private def checkTerminationConditions(): Boolean = {
    currentIteration >= maximumNumberOfIterations ||
    (checkForRankConvergence && lastMeanAbsoluteRankChange <= rankChangeTolerance)
  }

  private def performIterationUpdates() = {
    graph.computeVerticesUpdatedContent(vertexUpdateFunction)
    graph.updateVerticesContent()
    if (checkForRankConvergence) {
      lastMeanAbsoluteRankChange = 0.0
      for (vertex <- graph.vertices) lastMeanAbsoluteRankChange += vertex.content.lastAbsoluteChange
      lastMeanAbsoluteRankChange /= graph.numberOfVertices
    }
  }

  private def logIteration() = {
    if (checkForRankConvergence)
      logger.info(f"Iteration #: $currentIteration%10d | Mean Absolute Rank Change: $lastMeanAbsoluteRankChange%20s")
    else logger.info(f"Iteration #: $currentIteration%10d")
  }

  private def vertexUpdateFunction(vertex: Vertex[VertexContent, E]): VertexContent = {
    val oldRank: Double = vertex.content.rank
    var newRank: Double = 0.0
    vertex.incomingEdges.foreach(v => newRank += v.source.content.rank / v.source.numberOfOutgoingEdges)
    newRank *= dampingFactor
    newRank += dampingTerm
    VertexContent(
      vertex.content.id,
      newRank,
      if (checkForRankConvergence) Math.abs(newRank - oldRank) else 0.0
    )
  }
}

object PageRankAlgorithm {
  val logger = Logger(LoggerFactory.getLogger("Graph / PageRank"))

  case class VertexContent(id: Int, rank: Double, lastAbsoluteChange: Double = Double.MaxValue)
}
