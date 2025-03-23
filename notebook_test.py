import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
from numpy import mean, std
from notebook import df, dfDummies, dfNormalized, X, y, models

class TestNotebook(unittest.TestCase):

    def test_data_loading(self):
        self.assertFalse(df.empty, "Dataframe is empty")
        self.assertIn('class', df.columns, "Target column 'class' is missing")

    def test_dummies_creation(self):
        self.assertIn('protocol_type_tcp', dfDummies.columns, "Dummy columns not created correctly")
        self.assertNotIn('class', dfDummies.columns, "Target column 'class' should be dropped")

    def test_scaling(self):
        scaler = StandardScaler()
        scaler_df = scaler.fit_transform(dfDummies)
        dfNormalized = pd.DataFrame(scaler_df, columns=dfDummies.columns)
        self.assertAlmostEqual(dfNormalized.mean().mean(), 0, places=1, msg="Data not scaled correctly")
        self.assertAlmostEqual(dfNormalized.std().mean(), 1, places=1, msg="Data not scaled correctly")

    def test_model_evaluation(self):
        for name, model in models:
            kfold = KFold(n_splits=2, random_state=5, shuffle=True)
            cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
            self.assertGreater(mean(cv_results), 0.5, f"Model {name} accuracy is too low")
            self.assertLess(std(cv_results), 0.5, f"Model {name} accuracy variance is too high")

if __name__ == '__main__':
    unittest.main()
