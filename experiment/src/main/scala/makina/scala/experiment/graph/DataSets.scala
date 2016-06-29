package makina.scala.experiment.graph

import scala.collection.mutable
import scala.io.Source

/**
  * @author Emmanouil Antonios Platanios
  */
object DataSets {
  def loadUnlabeledSet(filePath: String, isDirected: Boolean): Set = {
    val name = filePath.split("/").last.split("\\.")(0)
    val dataSet = Set(name)
    for (line <- Source.fromFile(filePath).getLines) {
      val lineParts = line.split("\t")
      val sourceVertex = lineParts(0).toInt
      val destinationVertex = lineParts(1).toInt
      if (isDirected || !dataSet.edges.contains(Edge(destinationVertex, sourceVertex)))
        dataSet.addEdge(sourceVertex, destinationVertex)
    }
    dataSet
  }

  def loadLabeledSet(filePath: String, isDirected: Boolean): LabeledSet = {
    val name = filePath.split("/").last.split("\\.")(0)
    val dataSet = LabeledSet(name)
    val labelIds = mutable.Map[Int, Int]()
    for (line <- Source.fromFile(filePath + "/vertices.txt").getLines) {
      val lineParts = line.split("\t")
      val vertex = lineParts(0).toInt
      dataSet.addVertex(vertex)
      if (lineParts.length > 1) {
        val label = lineParts(1).toInt
        if (!labelIds.contains(label))
          labelIds += label -> labelIds.size
        dataSet.addVertexLabel(vertex, labelIds.get(label).get)
      }
    }
    for (line <- Source.fromFile(filePath + "/edges.txt").getLines) {
      val lineParts = line.split("\t")
      val sourceVertex = lineParts(0).toInt
      val destinationVertex = lineParts(1).toInt
      if (isDirected || !dataSet.edges.contains(Edge(destinationVertex, sourceVertex)))
        dataSet.addEdge(sourceVertex, destinationVertex)
    }
    dataSet
  }

  object Set {
    def apply(name: String) = new Set(name)
  }

  class Set(name: String) {
    val vertices = mutable.Set[Int]()
    val edges = mutable.Set[Edge]()

    def addVertex(vertex: Int) = vertices += vertex

    def addEdge(sourceVertex: Int, destinationVertex: Int) = {
      vertices += sourceVertex
      vertices += destinationVertex
      edges += Edge(sourceVertex, destinationVertex)
    }
  }

  object LabeledSet {
    def apply(name: String) = new LabeledSet(name)
  }

  class LabeledSet(val name: String) extends Set(name) {
    val vertexLabels = mutable.Map[Int, Int]()
    val vertexStatistics = mutable.Map[Int, Int]()

    def addVertexLabel(vertex: Int, label: Int) = {
      vertexLabels += vertex -> label
      if (!vertexStatistics.contains(label))
        vertexStatistics += label -> 0
      vertexStatistics(label) += 1
    }

    def getVertexLabel(vertex: Int) = vertexLabels(vertex)
  }

  case class Edge(sourceVertex: Int, destinationVertex: Int)
}
