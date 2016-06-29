package makina.scala.learn.neural.network

import breeze.linalg.DenseVector

import scala.collection.mutable

/**
  * TODO: Add support for variables initialization method.
  *
  * @author Emmanouil Antonios Platanios
  */
class NetworkState private (private val variablesManager: VariablesManager) {
  private val variables = mutable.Map[Variable, DenseVector[Double]]()
  for (variable <- variablesManager.variables)
    variables += variable -> DenseVector.zeros[Double](variable.size)

  def set(variable: Variable, value: DenseVector[Double]) = variables += variable -> value
  def set(id: Int, value: DenseVector[Double]) = variables += variablesManager.get(id) -> value
  def set(name: String, value: DenseVector[Double]) = variables += variablesManager.get(name) -> value

  def get(variable: Variable) = variables(variable)
  def get(id: Int) = variables(variablesManager.get(id))
  def get(name: String) = variables(variablesManager.get(name))
}

object NetworkState {
  def apply(variablesManager: VariablesManager) = new NetworkState(variablesManager)
}
