import os
from collections import OrderedDict

import pandas as pd
import torch

from anfis.membership_functions.trapezoidal_membership_function import TrapezoidalMembershipFunction
from datasets.weather_dataset import WeatherDataset
from utils import get_weather_anfis_model
import seaborn as sns
import matplotlib.pyplot as plt


def denormalize(value, std, mean):
    return (value.item() if isinstance(value, torch.Tensor) else value) * (std + 1e-7) + mean


os.makedirs("output/learnt_functions", exist_ok=True)

train_ds = WeatherDataset(f"data/weatherAUS_train.csv")

model = get_weather_anfis_model(pretrained_weights_filename="output/20220205_155404/model_69.pth")
for i, variable in enumerate(["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "Humidity3pm", "Pressure3pm"]):
    mfs = OrderedDict(
        [("low", model.layers["fuzzification"].variables[i].membership_functions['mf0']),
         ("medium", model.layers["fuzzification"].variables[i].membership_functions['mf1']),
         ("high", model.layers["fuzzification"].variables[i].membership_functions['mf2'])]
    )

    std = train_ds.stds[variable]
    mean = train_ds.means[variable]

    data = []

    print(f"{variable}:")
    for mf_name, mf in mfs.items():
        data.extend([
            {
                "universe": denormalize(mf.a, std, mean),
                "m": 0,
                "mf": mf_name
            },
            {
                "universe": denormalize(mf.b, std, mean),
                "m": 1,
                "mf": mf_name
            },
            {
                "universe": denormalize(mf.c, std, mean),
                "m": 1,
                "mf": mf_name
            },
            {
                "universe": denormalize(mf.d, std, mean),
                "m": 0,
                "mf": mf_name
            }
        ])

        repr = ""
        if mf.b == -TrapezoidalMembershipFunction.pseudo_infinity:
            repr += "-inf, "
        else:
            repr += f"{denormalize(mf.a, std, mean)}, {denormalize(mf.b, std, mean)}, "

        if mf.c == TrapezoidalMembershipFunction.pseudo_infinity:
            repr += "inf"
        else:
            repr += f"{denormalize(mf.c, std, mean)}, {denormalize(mf.d, std, mean)}"

        print(f"{mf_name}: ({repr})")

    df = pd.DataFrame(data)

    plt.figure(figsize=(3, 2))
    g = sns.lineplot(x=df["universe"], y=df["m"], hue=df["mf"])
    g.set_title(variable)
    g.set(xlim=(mean - 3*std, mean + 3*std))

    plt.savefig(f"output/learnt_functions/{variable}.png")
    plt.close()

pass
