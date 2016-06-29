package makina.scala.learn.neural.network

import java.util.concurrent.atomic.AtomicInteger

import breeze.linalg.DenseVector

import scala.collection.mutable

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Manager {
  protected val idCounter = new AtomicInteger(0)

  protected def id = idCounter.getAndIncrement()
}

case class VariablesManager() extends Manager {
  private val variablesById = mutable.Map[Int, Variable]()
  private val variablesByName = mutable.Map[String, Variable]()

  def variables = variablesById.values
  def get(id: Int) = variablesById(id)
  def get(name: String) = variablesByName(name)

  private def checkName(name: String) = {
    if (variablesByName.contains(name))
      throw new IllegalArgumentException("Each variable should have a unique name.")
  }

  private def addVariable[V <: Variable](variable: V) = {
    variablesById += variable.id -> variable
    variablesByName += variable.name -> variable
    variable
  }

  def constantVariable(value: DenseVector[Double]) = addVariable(ConstantVectorVariable(id, value))

  def vectorVariable(size: Int) = addVariable(VectorVariable(id, size))

  def vectorVariable(name: String, size: Int) = {
    checkName(name)
    addVariable(VectorVariable(id, name, size))
  }

  def matrixVariable(rowDimension: Int, columnDimension: Int) =
    addVariable(MatrixVariable(id, rowDimension, columnDimension))

  def matrixVariable(name: String, rowDimension: Int, columnDimension: Int) = {
    checkName(name)
    addVariable(MatrixVariable(id, name, rowDimension, columnDimension))
  }

  def layerVariable(layer: Layer) = addVariable(LayerVariable(id, layer))

  def layerVariable(name: String, layer: Layer) = {
    checkName(name)
    addVariable(LayerVariable(id, name, layer))
  }
}

case class LayersManager() extends Manager {
  private val layersById = mutable.Map[Int, Layer]()

  def layers = layersById.values
  def get(id: Int) = layersById(id)

  def add(layer: Layer) = {
    layersById += id -> layer
    layer
  }
}
