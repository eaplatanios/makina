package makina.scala.optimization

import breeze.linalg.DenseVector
import org.scalatest._

/**
  * @author Emmanouil Antonios Platanios
  */
class LBFGSQuasiNewtonSolverSpec extends FlatSpec {
  "The LBFGS quasi-Newton solver" should "be able to find the minimum of the Rosenbrock function" in {
    val function = new RosenbrockFunction
    val point = DenseVector(-1.2, 1.0)
    val solver = LBFGSQuasiNewtonSolver(logging = 5, lineSearch = NoLineSearch(1.0))
    val solution = solver.minimize(function, point)
    assert(solution === DenseVector.ones[Double](2))
  }
}
