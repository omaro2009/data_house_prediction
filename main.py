#importing the essential libraries for data analytics
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression ,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2,SelectFromModel,f_classif, VarianceThreshold,mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")

'''
House Price Prediction(Task1)
Use a dataset that includes information about
housing prices and features like square
footage, number of bedrooms, etc. to train a
model that can predict the price of a new
house

'''

location='F:\\internship-datascience\\3 cases\\House Prediction\\Code\\kc_house_data.csv'
class data_analysis:
    
    def __init__(self,location):
        self.location=location
        
    
    def data_analytics(self,location):
        '''
        This function is responsible for making the data ready
        for the prediction process
        Args:
        location is the location where the data set is found
        Returns:
        a ready set dataframe for prediction model
        '''
       
        self.df=pd.read_csv(location)
        print(self.df.shape)# the result of the shape is (21613, 21) which means there are 20 features
        def is_missing():
            '''
            checking if the dataframe first is having any missing data
            '''
            print(self.df.isnull().values.any())
            
        is_missing()# checking if any record having NaN field the answer is false
        def description():
            '''
            This function is passing all the important information about the general data for further data analytics
            '''
            #print(df.dtypes)
            '''
            id                 int64
            date              object
            price            float64
            bedrooms           int64
            bathrooms        float64
            sqft_living        int64
            sqft_lot           int64
            floors           float64
            waterfront         int64
            view               int64
            condition          int64
            grade              int64
            sqft_above         int64
            sqft_basement      int64
            yr_built           int64
            yr_renovated       int64
            zipcode            int64
            lat              float64
            long             float64
            sqft_living15      int64
            sqft_lot15         int64
            dtype: object
            '''
            self.df.drop(labels='id',axis=1,inplace=True)
            #print(self.df.dtypes) #dropped the id column as it is nonsense to carry our statical description having the column in the analysis
            '''
            date              object
            price            float64
            bedrooms           int64
            bathrooms        float64
            sqft_living        int64
            sqft_lot           int64
            floors           float64
            waterfront         int64
            view               int64
            condition          int64
            grade              int64
            sqft_above         int64
            sqft_basement      int64
            yr_built           int64
            yr_renovated       int64
            zipcode            int64
            lat              float64
            long             float64
            sqft_living15      int64
            sqft_lot15         int64
            dtype: object
            '''
            with open('statistics.txt',"w") as f:
                f.write(str(self.df.describe()))#writing the final summary of statistics in a txt file
                
                
        description()
        def exploratory_floor_count():
            ''' After being introduced to the data set knowing the all the important
                Statistics this function is used to explore all the feature properties
                to analyze the target feature floor which is the price
                
            '''
            
            floor_count=self.df['floors'].value_counts()
            
            '''
            floors
            1.0    10680
            2.0     8241
            1.5     1910
            3.0      613
            2.5      161
            3.5        8
            Name: count, dtype: int64
            '''
            floor_count.sort_index(inplace=True)
            print(floor_count)
            '''
            1.0    10680
            1.5     1910
            2.0     8241
            2.5      161
            3.0      613
            3.5        8
            '''
            def median_floor_count_price():
                '''
                This function return the mean price          '''
                floor_price_median_list=[]
                for i in floor_count.index:
                    floor_price=self.df.loc[self.df['floors']==i]
                    floor_price=floor_price['price']
                    floor_price_median=np.median(floor_price)
                    floor_price_median_list.append(floor_price_median)
                #print(floor_price_median_list)#[390000.0, 524475.0, 542950.0, 799200.0, 490000.0, 534500.0]
                def graph_floor_count():
                    '''
                    showing the graph that compares between floor and price
                    '''
                    X=self.df['floors']
                    Y=self.df["price"]
                    #print(Y)
                    sns.boxplot(x=X,y=Y,data=self.df)
                    plt.savefig('floor and price')
                    plt.show()
                    # there are many outliers coming from the number of floors and price
                graph_floor_count()
            def  graph_watervsprice():
                '''
                A boxplot graph to analyze the outliers of the waterfront feature
                '''
                X=self.df['waterfront']
                Y=self.df['price']
                sns.boxplot(x=X,y=Y,data=self.df)
            
                plt.savefig('waterfront vs price')
                plt.show()
            median_floor_count_price()
            graph_watervsprice()
        exploratory_floor_count()
        
    def Predicting_model(self):
        '''
        This function is responsible for performing prediction model input from data coming from
        The analysis stage. The 20 features will be filtered to have the most crucial features affecting the price
        
        Returns:
        Accuracy of model
        '''
        features=["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
        
         

        
        def Linear_model():
            '''
            First Model is the linear regression this function investigate the accuracy of the model
            '''
            lm=LinearRegression()
            X=self.df[features]
            Y=self.df['price']
            lm.fit(X,Y)
            print('The accuracy result of the linear model is '+str(lm.score(X,Y)))#0.6577024445001938
            
        def Pipelines():
            
            '''
            This function is intiating a pipeline which has three features
            1-Scaler --> Standard Scaler for pre-processing data.The main aim to transform the data to zero mean
            2-Feature selector--> there are two feature selector that will be tested
            -VarianceThreshold --> 
            -SelectFromModel--> This is used to select the feature importance’s from a model so that they can be used to train another model.
            3- Classifer --> First the linear regression will be tested again
            '''
            input_pipeline_1=[('scale',StandardScaler()),('selector', VarianceThreshold()),('model',LinearRegression())]
            input_pipeline_2=[('scale',StandardScaler()),('skb', SelectKBest(f_classif, k = 11)),('model',LinearRegression())]
            pipe1=Pipeline(input_pipeline_1)
            pipe2=Pipeline(input_pipeline_2)               
            X=self.df[features]
            Y=self.df['price']
            pipe1.fit(X,Y)
            pipe2.fit(X,Y)               
            print('The accuracy result of the linear model with variance threshold as feature selection technique is '+str(pipe1.score(X,Y)))#0.6576911310530571
            print('The accuracy result of the linear model with SelectKBest as feature selection technique is '+str(pipe2.score(X,Y)))#0.6576911310530571
            
            #Still the score isnot enhanced alot so the model must be changed
        def Lasso_Polynomial():
            'Linear regression has not suceeded to reach value higher than 66% so polynomial function will be replaced by linear regression'
            X=self.df[features]
            Y=self.df['price']
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
            cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
            # define model
            model = LassoCV(alphas=np.arange(0, 1, 0.1), cv=cv, n_jobs=-1)
            # fit model
            model.fit(x_test, y_test)
            print('The accuracy result of the Lasso model is '+str(model.score(x_test, y_test)))#0.6554377506441982
            pr = PolynomialFeatures(degree=2)
            x_train_pr = pr.fit_transform(x_train)
            x_test_pr = pr.fit_transform(x_test)
            model.fit(x_train_pr, y_train)
            model.score(x_train_pr, y_train)
        def Ridge_polynomial():
            'Finally Another regression algorithm is implemented with a new feature polynomial feature'
            X=self.df[features]
            Y=self.df['price']
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
            Ridge_test = Ridge(alpha = 0.0005)
            Ridge_test.fit(x_test, y_test)
            Ridge_test.score(x_test, y_test)
            pr = PolynomialFeatures(degree=2)
            x_train_pr = pr.fit_transform(x_train)
            x_test_pr = pr.fit_transform(x_test)
            Ridge_test.fit(x_train_pr, y_train)
            print('The accuracy of  ridge test implemented by polynomial feature is '+str(Ridge_test.score(x_train_pr, y_train)))#0.7423548624883267
            
            
        Linear_model()
        Pipelines()
        #modified_lasso()
        Ridge_polynomial()
dataframe=data_analysis(location)
dataframe.data_analytics(location)
dataframe.Predicting_model()
#dataframe.Predicting_model()
