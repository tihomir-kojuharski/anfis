from datetime import datetime
from enum import Enum
from math import inf

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from anfis.anfis_net import AnfisNet
from anfis.fuzzy_variable import FuzzyVariable
from anfis.membership_functions.boolean_membership_function import BooleanMembershipFunction
from anfis.membership_functions.gaussian_membership_function import GaussianMembershipFunction
from anfis.membership_functions.trapezoidal_membership_function import TrapezoidalMembershipFunction
from datasets.weather_dataset import WeatherDataset


class DS(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def main():
    datasets = {}
    data_loaders = {}

    for ds_name in [DS.TRAIN, DS.VALIDATION]:
        datasets[ds_name] = WeatherDataset(f"data/weatherAUS_{ds_name}.csv")
        data_loaders[ds_name] = DataLoader(datasets[ds_name], batch_size=64, shuffle=True)

    # variables = [
    #     FuzzyVariable(feature, GaussianMembershipFunction.get_functions(0.2, [0.2, 0.5, 0.7]))
    #     for feature in datasets["train"].feature_names
    #     if feature != "RainToday"
    # ]
    #
    # variables.extend([FuzzyVariable("RainToday", BooleanMembershipFunction.get_all())])
    variables = [
        FuzzyVariable("Rainfall", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("Pressure3pm", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("MaxTemp", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("Humidity3pm", TrapezoidalMembershipFunction.get_range_functions()),
        FuzzyVariable("RainToday", BooleanMembershipFunction.get_all()),
    ]

    model = AnfisNet(variables, 1, head_activation=torch.nn.Sigmoid())

    epochs = 10

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    best_vloss = inf

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")

        model.train(True)
        avg_loss = train_one_epoch(model, data_loaders["train"], optimizer, loss_fn, epoch, writer)
        model.train(False)

        running_vloss = 0.

        target_true = 0
        predicted_true = 0
        correct_true = 0
        for i, (features, labels) in enumerate(data_loaders["validation"]):
            outputs = model(features)

            vloss = loss_fn(outputs, labels)
            running_vloss += vloss

            predicted_classes = (outputs > 0.5).float()
            target_true += (labels == 1).sum()

            predicted_true += (predicted_classes).sum()

            correct_true += torch.sum(labels * predicted_classes)

        avg_vloss = running_vloss / (i + 1)
        print(f"Loss train: {avg_loss}, validation: {avg_vloss}")

        recall = correct_true / target_true
        precision = correct_true / predicted_true
        f1_score = 2 * precision * recall / (precision + recall)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1_score}")

        writer.add_scalars("Training vs Validatoin Loss", {
            "Training": avg_loss,
            "Validation": avg_vloss
        }, epoch + 1)

        writer.flush()

        if avg_vloss < best_vloss:
            model_path = 'output/model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)


def train_one_epoch(model, data_loader, optimizer, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, (features, labels) in enumerate(data_loader):
        optimizer.zero_grad()

        result = model(features)

        batch_loss = loss_fn(result, labels)
        batch_loss.backward()
        optimizer.step()

        running_loss += batch_loss

        if i % 100 == 99:
            last_loss = running_loss / 100
            print(f" batch {i} loss: {last_loss}")
            tb_x = epoch_index * len(data_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

    return last_loss


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
