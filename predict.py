import torch
from torch.utils.data import DataLoader

from datasets.weather_dataset import WeatherDataset
from utils import get_weather_anfis_model


def predict():
    model = get_weather_anfis_model("./output/model_20220130_172541_18")
    device = torch.device("cpu")

    model.to(device)
    model.train(False)

    ds = WeatherDataset("./data/weatherAUS_test.csv")

    sample = ds[8]
    features, label = sample

    x = features.unsqueeze(dim=0).to(device)

    result = model(x)

    print(result)
    pass


if __name__ == "__main__":
    predict()

