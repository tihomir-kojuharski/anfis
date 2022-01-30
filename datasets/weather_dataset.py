import pandas as pd
import torch.utils.data


class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, filename: str, transform=None, target_transform=None):
        super(WeatherDataset, self).__init__()
        self.__df = pd.read_csv(filename)

        boolean_columns = ["RainToday", "RainTomorrow"]
        for column in boolean_columns:
            self.__df[column] = (self.__df[column] == "Yes").astype('float32')

        self.__transform = transform
        self.__target_transform = target_transform

        # self.feature_names = ["MinTemp", "MaxTemp", "Rainfall", "WindGustDir", "WindGustSpeed", "WindDir9am",
        #                       "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
        #                       "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm", "RainToday", "latitude",
        #                       "longitude"]

        # self.feature_names = ["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed",
        #                       "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
        #                       "Pressure9am", "Pressure3pm", "RainToday"]

        self.feature_names = ["Rainfall", "Pressure3pm", "MaxTemp", "Humidity3pm",  "RainToday"]

        # self.feature_names = ["MinTemp", "MaxTemp"]

        self.means = {}
        self.stds = {}

        for column in self.feature_names:
            if column in boolean_columns:
                continue

            self.means[column] = self.__df[column].mean()
            self.stds[column] = self.__df[column].std()

            self.__df[column] = ((self.__df[column] - self.means[column]) / (self.stds[column] + 1e-7)).astype(
                'float32')

    def __len__(self):
        return len(self.__df)

    def __getitem__(self, idx):
        df_sample = self.__df.iloc[idx]

        features = torch.tensor([df_sample[column] for column in self.feature_names])
        label = df_sample["RainTomorrow"]

        if self.__transform:
            features = self.__transform(features)

        if self.__target_transform:
            label = self.__target_transform(label)

        return features, label
