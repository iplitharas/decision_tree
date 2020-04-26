from tools.logger import Logger, logged
from tools.helpers import write, restore
import numpy as np
import pandas as pd
from typing import List, Iterable, Tuple
import matplotlib.pyplot as plt
import datetime
import os


class DataHandler:
    def __init__(self, trips: List[dict], config: dict, saves_dir: str,
                 results_dir: str):
        """
        Main properties:
            data_sets : returns the training and the testing data set.
        Main methods:
            create_features: create features on the requested data set.
            evaluate: evaluates the data on a training model.
        :param trips: Raw data fetched from public servers
        :param config: Contains all the constants
        :param saves_dir:
        """
        self.trips = trips
        self.config = config
        self.saves_dir = saves_dir
        self.results_dir = results_dir
        self.date_format = "%Y-%m-%d"

    @property
    def relative_arrival_times(self) -> pd.DataFrame:
        """
        Tries to restore or re-calculate the relative arrival times
        """
        trips = None
        arrival_checkpoint = os.path.join(self.saves_dir,
                                          "relative_arrival_times")
        try:
            trips = restore(file_path=arrival_checkpoint)
        except FileNotFoundError:
            trips = self.calculate_relative_arrival_times()
            write(file_path=arrival_checkpoint, data=trips)
        finally:
            return trips

    @property
    def arrival_times(self) -> pd.DataFrame:
        """
        Returns the Final arrival times = Door to Door arrival times
        """
        door_df = None
        trips_df = self.relative_arrival_times
        door_door_checkpoint = os.path.join(self.saves_dir, "arrival_times")
        try:
            door_df = restore(file_path=door_door_checkpoint)
        except FileNotFoundError:
            door_df = self.calculate_arrival_times(trips_df=trips_df)
            write(file_path=door_door_checkpoint, data=door_df)
        finally:
            return door_df

    @property
    def data_sets(self, last_training_day: str = "2016-04-30") -> \
            Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Based on the last_training day it split the data into
        training , testing data set
        """
        training_df, testing_df = None, None
        training_checkpoint = os.path.join(self.saves_dir, "training")
        testing_checkpoint = os.path.join(self.saves_dir, "testing")
        try:
            training_df = restore(file_path=training_checkpoint)
            testing_df = restore(file_path=testing_checkpoint)
        except FileNotFoundError:
            training_df, testing_df = self._split_data_set(last_training_day)
            write(file_path=training_checkpoint, data=training_df)
            write(file_path=testing_checkpoint, data=testing_df)
        finally:
            return training_df, testing_df

    def evaluate(self, tree, data_set: pd.DataFrame):
        """
        This method evaluates the trained tree based on the requested data_set
        :param tree: The trained tree
        :param data_set: normally testing or evaluate training set
        :return: The mean value of the deviation from the estimated departure
        and the original.
        """
        dates_str = list(data_set.columns)
        features = self.create_features(dates_str)
        deviation = []
        errors = []
        for date_str in dates_str:
            predicted_departure = tree.predict(
                features=features.loc[date_str, :])
            lateness = data_set.loc[:, date_str]
            # from each -60 until 0 mins left from home we select the
            # predicted on from the model.
            label = lateness.loc[predicted_departure]
            if label > 0:
                Logger.logger.error(
                    f"Error predicted departure is:{predicted_departure} "
                    f"corresponding"
                    f" arrival for this departure is: {label}")
                errors.append(label)
            deviation.append(label)
            Logger.logger.debug(
                f"{date_str} estimated_departure {predicted_departure} "
                f"deviation {deviation[-1]}")

        accuracy = ((len(deviation) - len(errors)) / len(deviation)) * 100
        Logger.logger.info(
            f"Total errors are: {len(errors)}/{len(deviation)} accuracy is:"
            f"{accuracy}%\n")
        plt.plot(deviation, linestyle='none', marker='.')
        plt.ylabel('minutes late')
        plt.title(
            "Results on validation data set with {:.2f}% ".format(accuracy))
        plt.savefig(os.path.join(self.results_dir, "validations_results.pdf"),
                    format='pdf', dpi=1200)
        plt.show()
        return np.mean(np.abs(deviation))

    @logged
    def create_features(self, dates: List[str]) -> pd.DataFrame:
        """
        This method create the features for each day depending on:
        if it's workday or Saturday or Sunday,
        If it's winter autumn spring summer
        :return: The Features DataFrame on the requested dataset
        """
        features = []
        for date in dates:
            current_date = datetime.datetime.strptime(date,
                                                      self.date_format).date()
            current_week = current_date.weekday()
            current_month = current_date.month
            day_of_week = np.zeros(7)
            day_of_week[current_week] = 1
            month_of_year = np.zeros(12)
            month_of_year[current_month - 1] = 1
            seasons = np.zeros(4)
            if current_month <= 2:
                seasons[0] = 1
            elif current_month <= 5:
                seasons[1] = 1
            elif current_month <= 8:
                seasons[2] = 1
            elif current_month <= 11:
                seasons[3] = 1
            else:
                seasons[0] = 1

            feature_set = {
                'Saturday': day_of_week[5],
                'Sunday': day_of_week[6],
                'winter': seasons[0],
                'spring': seasons[1],
                'summer': seasons[2],
                'autumn': seasons[3]
            }
            features.append(feature_set)
        features = pd.DataFrame(data=features, index=dates)
        return features

    @logged
    def calculate_relative_arrival_times(self) -> pd.DataFrame:
        """
        transform the arrival time as a function of departure time.
        example:  relative to  target 9 ,if arrival time was 8 ->  60 mins
        Pandas dataFrame columns: date -> date time object
                                  departure time -> mins
                                  arrival time -> mins
        Each row the entry of the fetched data
        :return:pd.Dataframe
        """
        target_hour = self.config['target_hour']
        target_minute = self.config['target_minute']
        train_dep_min = self.config['train_dep_min']
        train_dep_max = self.config['train_dep_max']
        Logger.logger.debug(f"Cleaning data part...\n"
                            f"Target arrival at work hour:{target_hour}\n"
                            f"Target arrival at work minute:{target_minute}\n"
                            f"Train departure time(min) relative to arrival "
                            f"time:{train_dep_min}mins\n"
                            f"Train departure time(max) relative to arrival "
                            f"time:{train_dep_max}mins\n")
        minutes_per_hour = 60
        final_trips = []
        for trip in self.trips:
            relative_departure = minutes_per_hour * (
                    trip['departure'].hour - target_hour) + \
                                 trip['departure'].minute - target_minute
            relative_arrival = minutes_per_hour * (
                    trip["arrival"].hour - target_hour) + \
                               trip["arrival"].minute - target_minute

            if train_dep_min < relative_arrival <= train_dep_max:
                new_trip = {"departure": relative_departure,
                            "arrival": relative_arrival,
                            "date": trip["departure"].date()}
                final_trips.append(new_trip)
        Logger.logger.debug(f"Total trips founded :#{len(final_trips)}")
        trips_df = pd.DataFrame(final_trips)
        return trips_df

    @logged
    def calculate_arrival_times(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """
        :return:
        """
        door_arrivals = {}
        train_dep_min = self.config['train_dep_min']
        train_dep_max = self.config['train_dep_max']
        home_departure_station = self.config['home_departure_station']
        arrival_station_work = self.config['arrival_station_work']

        Logger.logger.debug(f"Cleaning data...\n"
                            f"Home to departure station :"
                            f"{home_departure_station} mins\n"
                            f"Arrival station to work :{arrival_station_work} "
                            f"mins\n"
                            f"Train departure time(min) relative to arrival "
                            f"time:{train_dep_min} mins\n"
                            f"Train departure time(max) relative to arrival "
                            f"time:{train_dep_max} mins\n")
        # create a new Data frame with minute by minute predictions
        for day in trips_df.loc[:, "date"].unique():
            date_str = day.strftime(self.date_format)
            trips_today = trips_df.loc[trips_df.loc[:, "date"] == day, :]
            door_arrival = np.zeros(train_dep_max - train_dep_min)
            for i_row, door_departure in enumerate(
                    np.arange(train_dep_min, train_dep_max)):
                # find the next train departure
                station_arrival = door_departure + home_departure_station
                try:
                    # find for each minute all the departures >= station arrival
                    idx = trips_today.loc[trips_today.loc[:,
                                          "departure"] >= station_arrival,
                                          "departure"].idxmin()
                    # append all of them + time from arrival station to work
                    door_arrival[i_row] = trips_today.loc[
                                              idx, "arrival"] + \
                                          arrival_station_work
                except Exception:
                    door_arrival[i_row] = np.nan
            # door arrival -> numpy array
            door_arrivals[date_str] = pd.Series(door_arrival,
                                                index=np.arange(train_dep_min,
                                                                train_dep_max))

        return pd.DataFrame(door_arrivals).fillna(value=30, inplace=False)

    @logged
    def _split_data_set(self, last_training_day) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        This method splits the data set -> Arrival times based on the last
        training day
        in training ,testing data set
        :param last_training_day:
        :return: Tuple[pd.DataFrame, pd.DataFrame]
        """
        training = []
        testing = []
        door_df = self.arrival_times
        last_training_day = datetime.datetime.strptime(last_training_day,
                                                       self.date_format)
        for date_str in door_df.columns:
            this_date = datetime.datetime.strptime(date_str, self.date_format)
            if this_date <= last_training_day:
                training.append(date_str)
            else:
                testing.append(date_str)
        training_df = door_df.loc[:, training]
        testing_df = door_df.loc[:, testing]
        return training_df, testing_df

    @staticmethod
    def custom_scatter(x: Iterable, y: Iterable) -> None:
        """
        Custom scatter plot for exploring the original data set
        @param x: iterable
        @param y: iterable
        @return:
        """
        plt.plot(x, y,
                 color="black",
                 marker=".",
                 linestyle="none",
                 alpha=.1,
                 )
        plt.show()
