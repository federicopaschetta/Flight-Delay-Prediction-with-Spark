from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.stat import Correlation
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor

import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Saves command line input files

if __name__ == "__main__":
    dataList = sys.argv[1:]

# Creates SparkSession
print("CREATE SPARKSESSION")
print()
spark = SparkSession.builder.appName('Big Data Project').getOrCreate()
spark.sparkContext.setLogLevel('WARN')


# Completes path to input data
dir_env_to_data = "./data/"
airport_file_path = 'airports.csv'
target_variable = 'ArrDelay'





# Takes all input files passed from command line, reads and merges them into one dataset
def merge_datasets(dataset_list):
    dfs = [spark.read.format('csv').option("delimeter", ",").option('header', 'true').load(dir_env_to_data+file_path) for file_path in dataList]
    df = dfs[0]
    for partial_df in dfs[1:]:
        df = df.union(partial_df)
    print('MERGED DF')
    return df

df = merge_datasets(dataList)
df.show(5, truncate=False, vertical=True)

print()
print("Data Cleaning")
print()


# Unifies all NA (strings in csv file) to None
for column in df.columns:
    df = df.withColumn(column, when(col(column)=="NA", None).otherwise(col(column)))
df.show(truncate=False, vertical=True)


# Deletes forbidden variables
print()
print("DELETING FORBIDDEN VARIABLES")
print()
deleting_attr = ['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 
                 'CarrierDelay', 'WeatherDelay', 'NASDelay', 
                 'SecurityDelay', 'LateAircraftDelay']
df = df.drop(*deleting_attr)
df.show(5, truncate=False, vertical=True)


# Computes null values percentage matrix
print()
print("COMPUTING NULL VALUES PERCENTAGE MATRIX")
print()
total_rows = df.count()
null_table = df.select([(count(when(col(c).isNull(), c))/(total_rows)*100).alias(c) for c in df.columns])
null_table.show(truncate=False, vertical=True)



# Checks if NA in response variable is 0
print()
print("CHECKING 0s IN RESPONSE VARIABLE")
print()
filtered_df = df.filter(col(target_variable) == 0)
if filtered_df.count() > 0:
    print("There are " + str(filtered_df.count()) + " 0s out of " + str(total_rows) + " in the ArrDelay column .")
else:
    print("There is not any 0 in the ArrDelay column.")


# Drops attributes with more than 25% of null values
print()
print("DROPPING ATTRIBUTES WITH MANY NULL VALUES")
print()
dropping_columns = [col_name for col_name in null_table.columns if null_table.filter(col(col_name) > 25).count() > 0]
null_table = null_table.drop(*dropping_columns)
df = df.drop(*dropping_columns)
print("Dropped "+ str(dropping_columns) + " columns.")
df.show(5, truncate=False, vertical=True)


# As null values % is low for every attribute, drop rows with NA
print()
print("DROPPING ROWS WITH NAs")
print()
df = df.dropna()
dropped_rows = total_rows - df.count()
print(str(dropped_rows)+ " rows have been dropped.")


# Drops variables with only one value in the whole dataset
print()
print("DROPPING UNIQUE VARIABLES")
print()
distinct_table = df.agg(*(countDistinct(col_name).alias(col_name) for col_name in df.columns)).limit(1)
distinct_table.show(truncate=False, vertical=True)

unique_columns =  [col_name for col_name in distinct_table.columns if distinct_table.filter(col(col_name) == 1).count() > 0]
df = df.drop(*unique_columns)
print(str(unique_columns)+ " rows have been dropped.")


# Checks for duplicates
print()
print("CHECKING DUPLICATES")
print()
total_rows = df.count()
df = df.dropDuplicates()

if total_rows - df.count()  > 0:
    print("There are", total_rows - df.count(), "duplicates in the DataFrame.")
else:
    print("There isn't any duplicate found in the DataFrame.")

print()
print("Variable transformation")
print()


# Reads airports dataset
print()
print("READING AIRPORTS CSV")
print()
airports_df = spark.read.format('csv').option("delimeter", ",").option('header', 'true').load(dir_env_to_data+airport_file_path)
airports_df = airports_df.select(['iata', 'lat', 'long'])
airports_df.show(5, truncate=False, vertical=True)


## Checks if there are missing airports 
# Collects IATA codes in airports df and in both origin and destination columns in main df
airports_df_iata = airports_df.select("iata").rdd.flatMap(lambda x: x).collect()
main_df_origin_iata = df.select("Origin").distinct().rdd.map(lambda x: x[0]).collect()
main_df_dest_iata = df.select("Dest").distinct().rdd.map(lambda x: x[0]).collect()

# Gets missing airports with set difference
print()
print("GETTING MISSING AIRPORTS")
print()
main_df_iata = set(main_df_origin_iata+main_df_dest_iata)
missing_airports = list(set(main_df_iata)-set(airports_df_iata))
print("There are ", len(missing_airports), " missing airports")

print()
print("JOINING DATASETS ON ORIGIN")
print()
# Joins datasets on Origin, renames new columns and drops old ones
df = df.join(airports_df, df.Origin == airports_df.iata, 'inner')
df = df.withColumnRenamed("lat", "OriginLat")
df = df.withColumnRenamed("long", "OriginLong")
df = df.drop("Origin")
df = df.drop("iata")

# Joins datasets on Destination, renames new columns and drops old ones
print()
print("JOINING DATASETS ON DESTINATION")
print()
df = df.join(airports_df, df.Dest == airports_df.iata, 'inner')
df = df.withColumnRenamed("lat", "DestLat")
df = df.withColumnRenamed("long", "DestLong")
df = df.drop("Dest")
df = df.drop("iata")


# Transform all numerical variables from string to int
print()
print("TRANSFORMING NUM VARIABLES")
print()
categorical_vars = ['UniqueCarrier']
for col_name in df.columns:
    if col_name not in categorical_vars:
        df = df.withColumn(col_name, col(col_name).cast('int'))
df.printSchema()


# Transforming times from int (with hhmm standard) to minutes
print()
print("TRANSFORMING TIMES TO MINUTES")
print()
def get_mins(hours):
    hours_val = (math.floor(hours/100))
    mins = hours-hours_val*100
    return hours_val*60+mins

# Calls previous function only for time variables and drops eventual wrong formatted rows
init_rows = df.count()
custom_udf = udf(get_mins, IntegerType())
hours_columns = ['DepTime', 'CRSDepTime', 'CRSArrTime']
for column in hours_columns:
    df = df.filter(~(col(column)>2359))
    df = df.withColumn(column, custom_udf(col(column)))


# Gets Number of categories for each categorical var
print()
print("COMPUTING CATEGORIES")
print()
for col_name in categorical_vars:
    print(col_name, ' has ', distinct_table.select(col_name).collect()[0][col_name],' different categories')


# Transforms "Origin" and "Dest" with coordinates





# Use One Hot Encoding for categorical columns
print()
print("INDEXING CATEGORICAL VARS")
print()

# Creates indexes for UniqueCarrier col and transforms df creating new col with the right index for each row
unique_carrier_indexer = StringIndexer(inputCol='UniqueCarrier', outputCol='UniqueCarrier_Index')
df = unique_carrier_indexer.fit(df).transform(df)

print()
print("ONE HOT ENCODING")
print()

# Encodes with OneHotEncoding UniqueCarrier based on col previously created and adds new OneHotEncoding column
unique_carrier_encoder = OneHotEncoder(inputCol='UniqueCarrier_Index', outputCol='UniqueCarrier_OneHot')
df = unique_carrier_encoder.fit(df).transform(df)


# Deletes original variables and indexes
deleting_cat_vars = []
for cols in categorical_vars:
    deleting_cat_vars.extend([cols, cols+'_Index'])
print("Categorical variables "+deleting_cat_vars+" need to be deleted")


# Renames column removing 'OneHot' from column name
df = df.drop(*deleting_cat_vars)
df = df.withColumnRenamed('UniqueCarrier_OneHot', 'UniqueCarrier')
print()
print("Final dataframe has ", len(df.columns), " columns")
print()
df.show(5, truncate=False, vertical=True)

print("DATA ANALYSIS")


'''
# Inspecting yearly (if multiple years in df) or monthly trends

# Selects grouping variable
if 'Year' in df.columns:
    grouping_var = 'Year'
else:
    grouping_var = 'Month'

# Groups ArrDelay values based on grouping var and gets values to plot
grouped_df = df.groupBy(grouping_var).agg(avg('ArrDelay').alias('avg'))
grouped_df.printSchema()
plot_values = (grouped_df.select([grouping_var,'avg']).collect())
plot_dict = {}

# Sorts values to plot based on month or year
for value in plot_values:
    plot_dict[value[grouping_var]] = value['avg']
sorted_dict = dict(sorted(plot_dict.items()))

# Plots values
plt.title('Average ArrDelay for '+ grouping_var)
plt.xlabel(grouping_var)
plt.ylabel('Delay (mins)')
plt.plot(sorted_dict.keys(), sorted_dict.values(),  marker='o')
plt.show()

'''

# Gets correlation matrix given the dataframe and numerical variables
def get_corr_matrix(df, numerical_vars):

    # Create a vector assembler to assemble the features
    assembler = VectorAssembler(inputCols=numerical_vars, outputCol="features")
    assembled_df = assembler.transform(df)

    # Calculate the correlation matrix
    corr_matrix = Correlation.corr(assembled_df, "features").head()[0]
    return corr_matrix


# Draws a heatmap based on correlation dataframe previously created
def draw_corr_heatmap(corr_df):
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths='.5')
    plt.title("Correlation Heatmap")
    plt.show()


# Detects variables with correlation > 0.9
def corr_feat_detect(corr_df):
    auto_feat_extraction = corr_df
    # Fills diagonal with NaN in order not to interfere in the detection
    np.fill_diagonal(auto_feat_extraction.values, np.nan)
    # Applies threshold condition to correlation matrix
    selected_cols = corr_df.columns[(corr_df.abs() > 0.9).any()]
    return selected_cols

# Gets numerical variables, computes correlation matrix and prints variables highly correlated
numerical_vars = list(set(df.columns)-set(categorical_vars))
corr_matrix = get_corr_matrix(df, numerical_vars)
corr_df = pd.DataFrame(corr_matrix.toArray(), columns=numerical_vars, index=numerical_vars)
#draw_corr_heatmap(corr_df, numerical_vars)
corr_cols_list = corr_feat_detect(corr_df)
print('Variables highly correlated: ', corr_cols_list)

# Removes some of highly correlated variables
df=df.drop(*['CRSElapsedTime', 'CRSDepTime'])


# Does the same process in order to detect any other high level of correlation
numerical_vars = list(set(df.columns)-set(categorical_vars))
corr_matrix = get_corr_matrix(df, numerical_vars)
corr_df = pd.DataFrame(corr_matrix.toArray(), columns=numerical_vars, index=numerical_vars)
#draw_corr_heatmap(corr_df, numerical_vars)
corr_cols_list = corr_feat_detect(corr_df)
print('Variables highly correlated: ', corr_cols_list)

# ## Modeling
print()
print("MODELING")
print()

# Scaling all numerical vars with MinMaxScaler
print()
print("SCALING VARIABLES")
print()

# For each numerical col (except target one) creates a vector with the values and scale them with MinMaxScaler
# transforming the dataframe by adding scaled values column
scaled_df = df
for column in df.columns:
    if column not in categorical_vars and column!=target_variable:
        assembler = VectorAssembler(inputCols=[column], outputCol=column+"_Vect")
        scaler = MinMaxScaler(inputCol=column+"_Vect", outputCol=column+"_Scaled")
        pipeline = Pipeline(stages=[assembler, scaler])
        scaled_df = pipeline.fit(scaled_df).transform(scaled_df)

# Drops old columns (both Vect and original ones)
for cols in scaled_df.columns:
    if cols not in categorical_vars and 'Scaled' not in cols and cols!=target_variable:
        scaled_df = scaled_df.drop(cols)
scaled_df.show(5, truncate=False, vertical=True)



scaled_df.show(5)


# Creates a list with columns to include in model just removing target variable
columns_to_include = scaled_df.columns
columns_to_include.remove('ArrDelay')
print('Columns to include: ', columns_to_include)



# Creates a vector features with all variable to include in model
model_assembler = VectorAssembler(inputCols=columns_to_include, outputCol="features")
scaled_df = model_assembler.transform(scaled_df)
print()


# Selects from dataframe only vector just created and target variable
modeling_df = scaled_df.select('features', 'ArrDelay')
print("Final Schema")
modeling_df.printSchema()

print()
print('Data Splitting')
print()

# Splits randomly train and test with a 80/20 proportion
train_data, test_data = modeling_df.randomSplit([0.8, 0.2], seed=123)


print()
print('LINEAR REGRESSION')
print()

print()
print('Linear Regression Model Creation')
print()
# Creates the linear regression model with features and prediction col, 
# then fits it to train data and adds predictions
# based on model to test data
lr = LinearRegression(featuresCol="features", labelCol="ArrDelay", predictionCol='Pred_ArrDelay')
lr_model = lr.fit(train_data)
lr_predictions = lr_model.transform(test_data)


print()
print('Linear Regression Evaluation')
print()

# Evaluates the model with metric RMSE (Root Mean Squared Errors), initializing RegressionEvaluator object
# and then applies it to predictions
rmse_lr_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="Pred_ArrDelay", metricName="rmse")
rmse_lr = rmse_lr_evaluator.evaluate(lr_predictions)
print()
print("Linear Regression - Root Mean Squared Error (RMSE) on test data: {:.2f}".format(rmse_lr))



# Evaluates the model with metric R^2, initializing RegressionEvaluator object
# and then applies it to predictions
r2_lr_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="Pred_ArrDelay", metricName="r2")
r2_lr = r2_lr_evaluator.evaluate(lr_predictions)
print()
print("Linear Regression - R-squared on test data =", r2_lr)

'''
# Creates a VectorAssembler to assemble all dataframe features
model_assembler = VectorAssembler(inputCols=columns_to_include, outputCol='features')

# Creates a LinearRegression model
lr = LinearRegression(featuresCol='features', labelCol='ArrDelay')

# Creates a pipeline with VectorAssembler and linear regression model
pipeline = Pipeline(stages=[model_assembler, lr])

# Creates an exploring parameters grid
paramgrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()

# Creates a RegressionEvaluator object to evaluate model performance            
evaluator = RegressionEvaluator(metricName='rmse', labelCol=lr.getLabelCol(), predictionCol=lr.getPredictionCol())

# Creates CrossValidator object
crossVal = CrossValidator(estimator=pipeline, estimatorParamMaps=paramgrid, 
                          evaluator=evaluator, numFolds=3)

                          
# Executes CrossValidator fitting to get results and best model
cvModel = crossVal.fit(scaled_df)
cvResults = cvModel.avgMetrics
bestModel = cvModel.bestModel
'''

# ## RandomForest
print()
print('RANDOM FOREST REGRESSION')
print()

print()
print('Random Forest Regression Model Creation')
print()
# Creates the random forest regression model with features, label col and an arbitrary number of trees, 
# then fits it to train data and adds predictions based on model to test data
rf = RandomForestRegressor(featuresCol="features", labelCol="ArrDelay", numTrees=100)
rf_model = rf.fit(train_data)
rf_predictions = rf_model.transform(test_data)



# Evaluates the model with metric RMSE (Root Mean Squared Errors), initializing RegressionEvaluator object
# and then applies it to predictions
rmse_rf_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="Pred_ArrDelay", metricName="rmse")
rmse_rf = rmse_rf_evaluator.evaluate(rf_predictions)
print("Random Forest Regression - Root Mean Squared Error (RMSE) on test data: {:.2f}".format(rmse_rf))


# Evaluates the model with metric R^2, initializing RegressionEvaluator object
# and then applies it to predictions
r2_rf_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="Pred_ArrDelay", metricName="r2")
r2_rf = r2_rf_evaluator.evaluate(rf_predictions)
print("Random Forest Regression - R-squared on test data =", r2_rf)





# Stops spark environment
spark.stop()

