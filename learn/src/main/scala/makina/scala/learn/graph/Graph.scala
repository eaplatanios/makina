package makina.scala.learn.graph

import scala.collection.mutable
import scala.collection.parallel

class Graph[V >: Null <: AnyRef, E >: Null <: AnyRef] private (isParallel: Boolean) {
  val vertices = mutable.Set[Vertex[V, E]]()
  val edges = mutable.Set[Edge[V, E]]()

  def numberOfVertices = vertices.size
  def numberOfEdges = edges.size

  def addVertex(vertex: Vertex[V, E]): Graph[V, E] = {
//    if (!vertices.contains(vertex))
    vertices += vertex
    this
  }

  def addEdge(edge: Edge[V, E]): Graph[V, E] = {
//    if (!vertices.contains(edge.source))
    vertices += edge.source
//    if (!vertices.contains(edge.destination))
    vertices += edge.destination
//    if (!edges.contains(edge))
    edges += edge
    this
  }

  def addEdge(source: Vertex[V, E], destination: Vertex[V, E], content: E = null): Graph[V, E] = {
//    if (!vertices.contains(source))
    vertices += source
//    if (!vertices.contains(destination))
    vertices += destination
    val edge = Edge(source, destination, content)
//    if (!edges.contains(edge))
    edges += edge
    this
  }

  def computeVerticesUpdatedContent(updateFunction: Vertex[V, E] => V) =
    if (isParallel)
      vertices.par.foreach(_.computeUpdatedContent(updateFunction))
    else
      vertices.foreach(_.computeUpdatedContent(updateFunction))

  def computeEdgesUpdatedContent(updateFunction: Edge[V, E] => E) =
    if (isParallel)
      edges.par.foreach(_.computeUpdatedContent(updateFunction))
    else
      edges.foreach(_.computeUpdatedContent(updateFunction))

  def computeUpdatedContent(vertexUpdateFunction: Vertex[V, E] => V,
                            edgeUpdateFunction: Edge[V, E] => E) = {
    computeVerticesUpdatedContent(vertexUpdateFunction)
    computeEdgesUpdatedContent(edgeUpdateFunction)
  }

  def updateVerticesContent() =
    if (isParallel)
      vertices.par.foreach(_.updateContent())
    else
      vertices.foreach(_.updateContent())

  def updateEdgesContent() =
    if (isParallel)
      edges.par.foreach(_.updateContent())
    else
      edges.foreach(_.updateContent())

  def updateContent() = {
    updateVerticesContent()
    updateEdgesContent()
  }
}

object Graph {
  def apply[V >: Null <: AnyRef, E >: Null <: AnyRef](isParallel: Boolean = true) = new Graph[V, E](isParallel)
}

/**
  * @tparam V Vertex content type.
  * @tparam E Edge content type.
  *
  * @author Emmanouil Antonios Platanios
  */
class Vertex[V >: Null <: AnyRef, E >: Null <: AnyRef] private (private var _content: V) {
  private var _updatedContent: V = null

  val incomingEdges = mutable.ArrayBuffer[Edge[V, E]]()
  val outgoingEdges = mutable.ArrayBuffer[Edge[V, E]]()

  def numberOfIncomingEdges = incomingEdges.size
  def numberOfOutgoingEdges = outgoingEdges.size
  def content: V = _content
  def updatedContent: V = _updatedContent

  def addIncomingEdge(edge: Edge[V, E]): Vertex[V, E] = {
    if (edge.destination != this)
      throw new IllegalArgumentException("The destination of the provided edge is not the current vertex.")
    incomingEdges += edge
    this
  }

  def addOutgoingEdge(edge: Edge[V, E]): Vertex[V, E] = {
    if (edge.source != this)
      throw new IllegalArgumentException("The source of the provided edge is not the current vertex.")
    outgoingEdges += edge
    this
  }

  def computeUpdatedContent(updateFunction: Vertex[V, E] => V) = _updatedContent = updateFunction(this)
  def updateContent() = _content = _updatedContent
}

object Vertex {
  def apply[V >: Null <: AnyRef, E >: Null <: AnyRef](content: V = null) = new Vertex[V, E](content)
}

/**
  * @tparam V Vertex content type.
  * @tparam E Edge content type.
  *
  * @author Emmanouil Antonios Platanios
  */
class Edge[V >: Null <: AnyRef, E >: Null <: AnyRef] private (val source: Vertex[V, E],
                                                              val destination: Vertex[V, E],
                                                              private var _content: E) {
  private var _updatedContent: E = null
  source.addOutgoingEdge(this)
  destination.addIncomingEdge(this)

  def content: E = _content
  def updatedContent: E = _updatedContent

  def computeUpdatedContent(updateFunction: Edge[V, E] => E) = _updatedContent = updateFunction(this)
  def updateContent() = _content = _updatedContent
}

object Edge {
  def apply[V >: Null <: AnyRef, E >: Null <: AnyRef](source: Vertex[V, E],
                                                      destination: Vertex[V, E],
                                                      content: E = null) =
    new Edge[V, E](source, destination, content)
}
