from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from datasets.weather_dataset import WeatherDataset
from utils import get_weather_anfis_model


def predict():
    model = get_weather_anfis_model("./output/20220205_155404/model_69.pth")
    device = torch.device("cpu")

    model.to(device)
    model.train(False)

    x = OrderedDict([
        ("MinTemp", 10),
        ("MaxTemp", 10),
        ("Rainfall", 50),
        ("WindGustSpeed", 10),
        ("Humidity3pm", 50),
        ("Pressure3pm", 1030),
        ("RainToday", 1)])

    train_ds = WeatherDataset("./data/weatherAUS_train.csv")
    features = []
    for feature in x:
        if feature == "RainToday":
            normalized_value = x[feature]
        else:
            normalized_value = (x[feature] - train_ds.means[feature]) / (train_ds.stds[feature] + 1e-7)
        features.append(normalized_value)

    features = torch.tensor(features, dtype=torch.float)

    test_ds = WeatherDataset("./data/weatherAUS_test.csv")
    sample = test_ds[8]
    # features, label = sample

    x = features.unsqueeze(dim=0).to(device)

    result = model(x)

    print(result)
    pass


if __name__ == "__main__":
    predict()

