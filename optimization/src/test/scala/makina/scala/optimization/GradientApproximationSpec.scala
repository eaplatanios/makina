package makina.scala.optimization

import breeze.linalg.DenseVector
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class GradientApproximationSpec extends FlatSpec {
  "ForwardDifference" should "estimate the gradient" in {
    val function = new RosenbrockFunction
    val point = DenseVector(-1.2, 1.0)
    assert(ForwardDifferenceGradientApproximation.gradient(function, point) === function.gradient(point))
  }
}
