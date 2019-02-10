package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._

/**
 * Contains logic for computing the trx_summary_features featuregroup
 */
object TrxSummaryFeatures {

  case class RawTrx(
    cust_id_in: Int,
    cust_id_out: Int,
    trx_amount: Double,
    trx_bankid: Int,
    trx_clearinnum: Int,
    trx_country: String,
    trx_date: String,
    trx_type: String,
    trx_id: Int)

  case class TrxSummaryFeature(
    cust_id: Long,
    min_trx: Float,
    max_trx: Float,
    avg_trx: Float,
    count_trx: Long)

  case class ParsedTrx(
    cust_id: Int,
    amount: Float)

  /**
   * Computes the featuregroup
   *
   * @param input path to the input dataset to read (csv)
   * @param output name of the output featuregroup table
   * @param version version of the featuregroup
   * @param partitions number of spark partitions to parallelize the compute on
   * @param logger spark logger
   * @param create boolean flag whether to create new featuregroup or insert into an existing ones
   * @param jobId the id of the hopsworks job
   */
  def computeFeatures(spark: SparkSession, input: String, featuregroupName: String,
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int): Unit = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input).repartition(partitions)
    log.info("Read raw dataframe")
    import spark.implicits._
    val rawDs = rawDf.as[RawTrx]
    val inTrx = rawDs.map((trx: RawTrx) => {
      ParsedTrx(trx.cust_id_in, trx.trx_amount.toFloat)
    })
    val outTrx = rawDs.map((trx: RawTrx) => {
      ParsedTrx(trx.cust_id_in, trx.trx_amount.toFloat)
    })
    val allTrx = inTrx.unionAll(outTrx)
    val min = allTrx.groupBy("cust_id").min("amount")
    val max = allTrx.groupBy("cust_id").max("amount")
    val sum = allTrx.groupBy("cust_id").sum("amount")
    val count = allTrx.groupBy("cust_id").count
    val rawFeaturesDf = min.join(max, "cust_id").join(count, "cust_id").join(sum, "cust_id")
    val features = rawFeaturesDf.rdd.map((row: Row) => {
      val min = row.getAs[Float]("min(amount)")
      val max = row.getAs[Float]("max(amount)")
      val sum = row.getAs[Double]("sum(amount)")
      val count = row.getAs[Long]("count")
      val avg = sum.toFloat / count.toFloat
      val custId = row.getAs[Int]("cust_id")
      new TrxSummaryFeature(custId.toLong, min, max, avg, count)
    }).toDS
    val featurestore = Hops.getProjectFeaturestore
    val descriptiveStats = true
    val featureCorr = true
    val featureHistograms = true
    val clusterAnalysis = true
    val statColumns = List[String]().asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "Aggregate of transactions for customers"
    val primaryKey = "cust_id"
    val dependencies = List[String](input).asJava
    if (create) {
      log.info(s"Creating featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.createFeaturegroup(spark, features.toDF, featuregroupName, featurestore, version, description,
        jobId, dependencies, primaryKey, descriptiveStats, featureCorr, featureHistograms, clusterAnalysis,
        statColumns, numBins, corrMethod, numClusters)
      log.info(s"Creation of featuregroup $featuregroupName complete")
    } else {
      log.info(s"Inserting into featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.insertIntoFeaturegroup(spark, features.toDF, featuregroupName, featurestore, version, "overwrite",
        descriptiveStats, featureCorr, featureHistograms, clusterAnalysis, statColumns, numBins, corrMethod,
        numClusters)
      log.info(s"Insertion into featuregroup $featuregroupName complete")
    }
  }
}
