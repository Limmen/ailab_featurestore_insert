package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.sql.SparkSession
import java.util.Date
import java.sql.Timestamp
import org.apache.spark.sql.Row
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._

/**
 * Contains logic for computing the demographic_features featuregroup
 */
object DemographicFeatures {

  case class RawParty(
    balance: Double,
    birthdate: String,
    customer_type: String,
    gender: String,
    join_date: String,
    name: String,
    number_of_accounts: Int,
    pep: Boolean,
    cust_id: Int)

  case class ParsedParty(
    balance: Float,
    birthdate: Timestamp,
    customer_type: Long,
    gender: Long,
    join_date: Timestamp,
    number_of_accounts: Int,
    pep: Long,
    cust_id: Int)

  val formatter = new java.text.SimpleDateFormat("yyyy-MM-dd")

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
   * @param customerTypeLookup optional pre-computed map with customer types categorical to numeric encoding
   * @param genderLookup optional pre-computed map with gender types categorical to numeric encoding
   * @param pepLookup optional pre-computed map with pep types categorical to numeric encoding
   */
  def computeFeatures(spark: SparkSession, input: String, featuregroupName: String,
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int, customerTypeLookup: Map[String, Long] = null,
    genderLookup: Map[String, Long] = null, pepLookup: Map[Boolean, Long] = null): Unit = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input).repartition(partitions)
    log.info("Read raw dataframe")
    log.info("Converting strings into dates")
    import spark.implicits._
    val rawDs = rawDf.as[RawParty]
    log.info("Parsed dataframe to dataset")
    log.info("Reading customer types lookup table from featurestore..")
    val featurestore = Hops.getProjectFeaturestore
    var customerTypeLookupMap: Map[String, Long] = null
    if (customerTypeLookup == null) {
      val customerTypeLookupDf = Hops.getFeaturegroup(spark, "customer_type_lookup", featurestore, 1)
      val customerTypeLookupList: Array[Row] = customerTypeLookupDf.collect
      customerTypeLookupMap = customerTypeLookupList.map((row: Row) => {
        row.getAs[String]("customer_type") -> row.getAs[Long]("id")
      }).toMap
    } else {
      customerTypeLookupMap = customerTypeLookup
    }
    var genderLookupMap: Map[String, Long] = null
    if (genderLookup == null) {
      val genderLookupDf = Hops.getFeaturegroup(spark, "gender_lookup", featurestore, 1)
      val genderLookupList: Array[Row] = genderLookupDf.collect
      genderLookupMap = genderLookupList.map((row: Row) => {
        row.getAs[String]("gender") -> row.getAs[Long]("id")
      }).toMap
    } else {
      genderLookupMap = genderLookup
    }
    var pepLookupMap: Map[Boolean, Long] = null
    if (pepLookup == null) {
      val pepLookupDf = Hops.getFeaturegroup(spark, "pep_lookup", featurestore, 1)
      val pepLookupList: Array[Row] = pepLookupDf.collect
      pepLookupMap = pepLookupList.map((row: Row) => {
        val pep = row.getAs[String]("pep")
        pep match {
          case "True" => true -> row.getAs[Long]("id")
          case "False" => false -> row.getAs[Long]("id")
        }
      }).toMap
    } else {
      pepLookupMap = pepLookup
    }
    val parsedDs = rawDs.map((party: RawParty) => {
      val balance = party.balance.toFloat
      val birthdate = new Timestamp(formatter.parse(party.birthdate).getTime)
      val customerType = customerTypeLookupMap(party.customer_type)
      val gender = genderLookupMap(party.gender)
      val joinDate = new Timestamp(formatter.parse(party.join_date).getTime)
      val numberOfAccounts = party.number_of_accounts
      val pep = pepLookupMap(party.pep)
      val cust_id = party.cust_id
      new ParsedParty(balance, birthdate, customerType, gender, joinDate, numberOfAccounts, pep: Long, cust_id)
    })
    val descriptiveStats = true
    val featureCorr = true
    val featureHistograms = true
    val clusterAnalysis = true
    val statColumns = List[String]().asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "preprocessed features from the KYC table"
    val primaryKey = "cust_id"
    val dependencies = List[String](input).asJava
    if (create) {
      log.info(s"Creating featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.createFeaturegroup(spark, parsedDs.toDF, featuregroupName, featurestore, version, description,
        jobId, dependencies, primaryKey, descriptiveStats, featureCorr, featureHistograms, clusterAnalysis,
        statColumns, numBins, corrMethod, numClusters)
      log.info(s"Creation of featuregroup $featuregroupName complete")
    } else {
      log.info(s"Inserting into featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.insertIntoFeaturegroup(spark, parsedDs.toDF, featuregroupName, featurestore, version, "overwrite",
        descriptiveStats, featureCorr, featureHistograms, clusterAnalysis, statColumns, numBins, corrMethod,
        numClusters)
      log.info(s"Insertion into featuregroup $featuregroupName complete")
    }
  }
}
