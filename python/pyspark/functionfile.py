from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Define a function to perform linear regression and return the evaluation metrics
def linear_regression(df, target_col, feature_cols):
    # Create a SparkSession
    spark = SparkSession.builder.appName("LinearRegression").getOrCreate()
    
    # Convert the pandas DataFrame to a Spark DataFrame
    sdf = spark.createDataFrame(df)
    
    # Split the dataset into training and testing sets
    trainingData, testData = sdf.randomSplit([0.7, 0.3])

    # Create a VectorAssembler object to combine all input columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Apply the VectorAssembler transformation to the training and testing data
    trainingData = assembler.transform(trainingData)
    testData = assembler.transform(testData)

    # Create a LinearRegression object
    lr = LinearRegression(featuresCol="features", labelCol=target_col)

    # Fit the model to the training data
    model = lr.fit(trainingData)

    # Make predictions on the testing data
    predictions = model.transform(testData)

    # Calculate the TSS and RSS
    TSS = sdf.select(col(target_col)).rdd.map(lambda x: x[0]).variance() * (sdf.count() - 1)
    RSS = predictions.select(col(target_col), col("prediction")).rdd.map(lambda x: (x[0] - x[1])**2).sum()

    # Calculate the R-squared value
    
    R_squared =1 - (RSS / TSS)

    # Print model coefficients and intercept
    coefficients = model.coefficients
    intercept = model.intercept
    print("Coefficients: ")
    feature_names = assembler.getInputCols()
    for i in range(len(coefficients)):
        print(feature_names[i], ": ", coefficients[i], " (p-value: ", model.summary.pValues[i], ")")
    print("Intercept: ", intercept)

    # Calculate and print MSE and RMSE
    evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction")
    mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    # Print evaluation metrics
    print("TSS: ", TSS)
    print("RSS: ", RSS)
    print("R-squared: ", R_squared)  

    # Stop the SparkSession
    spark.stop()

    
