package makina.scala.optimization

import breeze.linalg.DenseVector
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class SolverSpec extends FlatSpec {
  "The gradient descent solver" should "be able to find the minimum of the Rosenbrock function" in {
    val function = new RosenbrockFunction
    val point = DenseVector(-1.2, 1.0)
    val solver = GradientDescentSolver(logging = 5, lineSearch = BacktrackingLineSearch(contraptionFactor = 0.5, c = 0.5))
    val solution = solver.minimize(function, point)
    assert(solution === DenseVector.ones[Double](2))
  }
}
