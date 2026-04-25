import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class BoomCrashModel:
    def __init__(self):
        self.boom_years = [1926, 1936, 1945, 1953, 1962, 1972, 1980, 1989, 1999, 2007, 2014, 2026, 2034, 2043, 2053]
        self.crash_years = [1924, 1931, 1942, 1951, 1958, 1969, 1978, 1985, 1996, 2005, 2012, 2023, 2030, 2039, 2050, 2059]

        current_year = pd.Timestamp.now().year
        for year in range(1927, current_year + 50, 18):
            if year not in self.boom_years:
                self.boom_years.append(year)
            if year + 2 not in self.crash_years:
                self.crash_years.append(year + 2)

        self.boom_years.sort()
        self.crash_years.sort()

    def get_market_phase(self, date):
        year = date.year
        for boom_year in self.boom_years:
            if year >= boom_year and year <= boom_year + 2:
                return "boom"
        for crash_year in self.crash_years:
            if year >= crash_year and year <= crash_year + 2:
                return "crash"
        return "neutral"


def train_boom_crash_model():
    print("Creating boom/crash cycle model...")
    model = BoomCrashModel()
    joblib.dump(model, "boom_crash_model.pkl")
    print("Boom/crash model saved as boom_crash_model.pkl")


if __name__ == "__main__":
    train_boom_crash_model()