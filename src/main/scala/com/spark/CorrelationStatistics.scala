package com.spark

import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.mllib.stat.Statistics
/**
  * Created by sai.luo on 2018-2-7.
  */
object CorrelationStatistics {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("CorrelationExample").master("local")
      .getOrCreate()


    val data = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
    )

    import spark.implicits._
    val df = data.map(Tuple1.apply).toDF("features")
    val Row(coeff1: Matrix) = Correlation.corr(df, "features").head()
    println("Pearson correlation matrix:\n" + coeff1.toString)

    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
    println("Spearman correlation matrix:\n" + coeff2.toString)

    val  sc = spark.sparkContext
    //==================================计算相关性系数===================================
    val rdd2 = sc.parallelize(Array(1.0,2.0,3.0,4.0))
    val rdd3 = sc.parallelize(Array(2.0,3.0,4.0,5.0))
/*    val rdd2 = sc.parallelize(Array(149.0,150.0,153.0,155.0,160.0,155.0,160.0,150.0))
    val rdd3 = sc.parallelize(Array(81.0,88.0,87.0,99.0,91.0,89.0,95.0,90.0))*/
    println("rdd2:"+rdd2)
    rdd2.collect().foreach(println)
    //val correlation1:Double = Statistics.corr(rdd2, rdd3, "pearson")
    //省却的情况下，默认的是pearson相关性系数
    val correlation1:Double = Statistics.corr(rdd2, rdd3)
    println("pearson相关系数："+correlation1)
    //pearson相关系数：0.6124030566141675
    val correlation2:Double = Statistics.corr(rdd2, rdd3, "spearman")
    println("spearman相关系数："+correlation2)
    //spearman相关系数：0.7395161835775294
    sc.stop()
  }
}
