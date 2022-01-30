import pandas as pd
from sklearn.model_selection import train_test_split


def split():
    df = pd.read_csv("./data/weatherAUS_clean.csv")
    train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df["RainTomorrow"])

    train, validation = train_test_split(train, test_size=0.25, shuffle=True, random_state=42, stratify=train["RainTomorrow"])

    train.to_csv("./data/weatherAUS_train.csv", index=False)
    validation.to_csv("./data/weatherAUS_validation.csv", index=False)
    test.to_csv("./data/weatherAUS_test.csv", index=False)
    pass

if __name__ == "__main__":
    split()