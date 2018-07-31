import java.io.File

import breeze.linalg._

val file = new File("./data/binary.csv")
val mat = csvread(file, skipLines = 1)
val X = DenseMatrix.horzcat[DenseMatrix[Double], Double](DenseMatrix.ones[Double](mat(::, 1 to 3).rows, 1), mat(::, 1 to 3))
val y = mat(::, 0)

def u(w: DenseVector[Double]): DenseVector[Double] = X(*, ::).map(x =>  1/(1 + Math.exp(-(w dot x))))

def S(w: DenseVector[Double]) = diag(u(w).map(e => e * (1.0 - e)))

def step(Sk: DenseMatrix[Double], uk: DenseVector[Double], wk: DenseVector[Double]): DenseVector[Double] =
  inv(X.t * Sk * X) * X.t * (Sk * X * wk + y - uk)

def Pr(y: Int, x: DenseVector[Double], w: DenseVector[Double]) = {
  def h(x: DenseVector[Double], w: DenseVector[Double]) = 1/(1 + Math.exp(-(w dot x)))
  y match {
    case 0 =>
      1 - h(x, w)
    case 1 =>
      h(x, w)
  }
}
def likelyhood(w: DenseVector[Double]) = (0 until X.rows).map(i =>
  Pr(if(y(i) > 0.0) 1 else 0, X(i, ::).t, w)
).foldLeft(1.0)((s, p) => s * p)

var w = DenseVector(0.0, 0.0, 0.0, 0.0)
var wNext = step(S(w), u(w), w)
val tolerance = 0.0001
var counter = 0

while(norm(w - wNext) > tolerance && counter < 100){
  w = wNext
  wNext = step(S(w), u(w), w)

  while(likelyhood(w) > likelyhood(wNext)){
    wNext = w + ((w - wNext) /:/ 2.0)
  }

  counter += 1
}

s"counter = $counter, likelyhood(w) = ${likelyhood(w)}"
w
