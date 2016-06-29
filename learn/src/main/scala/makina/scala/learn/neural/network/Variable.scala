package makina.scala.learn.neural.network

import breeze.linalg.{DenseMatrix, DenseVector}
import makina.scala.learn.neural.network

/**
  * @author Emmanouil Antonios Platanios
  */
abstract class Variable {
  val id: Int
  val name: String
  val size: Int

  def value(state: NetworkState): DenseVector[Double]
}

case class ConstantVectorVariable(id: Int, value: DenseVector[Double]) extends Variable {
  override val name = id.toString
  override val size = value.size

  override def value(state: NetworkState): DenseVector[Double] = value
}

case class VectorVariable(id: Int, name: String, size: Int) extends Variable {
  override def value(state: NetworkState): DenseVector[Double] = state.get(this)
}

object VectorVariable {
  def apply(id: Int, size: Int): VectorVariable = VectorVariable(id, id.toString, size)
}

case class MatrixVariable(id: Int, name: String, rowDimension: Int, columnDimension: Int)
  extends Variable {
  override val size = rowDimension * columnDimension

  // TODO: Make this function return either matrices or vectors, depending on the type of variable.
  override def value(state: NetworkState): DenseVector[Double] = state.get(this)
  def valueInMatrixForm(state: NetworkState): DenseMatrix[Double] =
    new DenseMatrix[Double](rowDimension, columnDimension, state.get(this).toArray)
}

object MatrixVariable {
  def apply(id: Int, rowDimension: Int, columnDimension: Int): network.MatrixVariable =
    MatrixVariable(id, id.toString, rowDimension, columnDimension)
}

case class LayerVariable(id: Int, name: String, layer: Layer) extends Variable {
  override val size = layer.outputSize

  override def value(state: NetworkState): DenseVector[Double] = layer.value(state)
}

object LayerVariable {
  def apply(id: Int, layer: Layer): LayerVariable = LayerVariable(id, id.toString, layer)
}
