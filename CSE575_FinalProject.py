import pandas as pd
import warnings
import ComprehensiveRegressor as CR
warnings.filterwarnings('ignore')


train = pd.read_csv("allPMData.csv")

model =CR.ComprehensiveRegressor(train, 5, ['cb','xgb','gbm'])

model.train()
model.predict()
