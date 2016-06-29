package makina.scala.learn.graph

import makina.scala.learn.graph.PageRankAlgorithm.VertexContent
import org.scalatest._

import scala.util.Random

/**
  * @author Emmanouil Antonios Platanios
  */
class PageRankAlgorithmSpecification[E >: Null <: AnyRef] extends FlatSpec with ShouldMatchers {
  private val random: Random = new Random

  "PageRank with three non-linked pages " should "run" in {
    val graph = Graph[VertexContent, E]()
    val pageA = Vertex[VertexContent, E](VertexContent(0, random.nextDouble()))
    val pageB = Vertex[VertexContent, E](VertexContent(1, random.nextDouble()))
    val pageC = Vertex[VertexContent, E](VertexContent(2, random.nextDouble()))
    graph.addVertex(pageA)
    graph.addVertex(pageB)
    graph.addVertex(pageC)
    val algorithm = PageRankAlgorithm(graph, dampingFactor = 0.85, maximumNumberOfIterations = 1000, loggingLevel = 4)
    algorithm.computeRanks()
    0.05 should equal (pageA.content.rank +- 1e-2)
    0.05 should equal (pageB.content.rank +- 1e-2)
    0.05 should equal (pageC.content.rank +- 1e-2)
  }

  "PageRank with three fully linked pages " should "run" in {
    val graph = Graph[VertexContent, E]()
    val pageA = Vertex[VertexContent, E](VertexContent(0, random.nextDouble()))
    val pageB = Vertex[VertexContent, E](VertexContent(1, random.nextDouble()))
    val pageC = Vertex[VertexContent, E](VertexContent(2, random.nextDouble()))
    graph.addVertex(pageA)
    graph.addVertex(pageB)
    graph.addVertex(pageC)
    graph.addEdge(pageA, pageB)
    graph.addEdge(pageB, pageA)
    graph.addEdge(pageA, pageC)
    graph.addEdge(pageC, pageA)
    graph.addEdge(pageB, pageC)
    graph.addEdge(pageC, pageB)
    val algorithm = PageRankAlgorithm(graph, dampingFactor = 0.85, maximumNumberOfIterations = 1000, loggingLevel = 4)
    algorithm.computeRanks()
    0.33 should equal (pageA.content.rank +- 1e-2)
    0.33 should equal (pageB.content.rank +- 1e-2)
    0.33 should equal (pageC.content.rank +- 1e-2)
  }

  "PageRank with three partially linked pages " should "run" in {
    val graph = Graph[VertexContent, E]()
    val pageA = Vertex[VertexContent, E](VertexContent(0, random.nextDouble()))
    val pageB = Vertex[VertexContent, E](VertexContent(1, random.nextDouble()))
    val pageC = Vertex[VertexContent, E](VertexContent(2, random.nextDouble()))
    graph.addVertex(pageA)
    graph.addVertex(pageB)
    graph.addVertex(pageC)
    graph.addEdge(pageA, pageB)
    graph.addEdge(pageB, pageA)
    graph.addEdge(pageA, pageC)
    graph.addEdge(pageC, pageA)
    val algorithm = PageRankAlgorithm(graph, dampingFactor = 0.85, maximumNumberOfIterations = 1000, loggingLevel = 4)
    algorithm.computeRanks()
    0.487 should equal (pageA.content.rank +- 1e-3)
    0.257 should equal (pageB.content.rank +- 1e-3)
    0.257 should equal (pageC.content.rank +- 1e-3)
  }
}
