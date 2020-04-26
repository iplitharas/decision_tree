import json
import os
import requests
from requests.exceptions import HTTPError
from zipfile import ZipFile
from typing import List, Tuple, Text
import pandas as pd
import datetime
from tools.helpers import write, restore
from tools.logger import Logger, logged


class DataLoader:

    def __init__(self, saves_folder_name: str = "saves", results_folder_name: str = "results", debug=True):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_path = self.current_dir
        self.debug = debug
        self._config = self._read_config()
        self._saves_dir = None
        self._result_dir = None
        self._make_directories(saves_folder_name=saves_folder_name,
                               results_folder_name=results_folder_name)

    @property
    def config(self) -> dict:
        return self._config

    @property
    def saves_dir(self) -> Text:
        return self._saves_dir

    @property
    def data(self) -> List[dict]:
        """
        This method is the entry point to retrieve all the data set from (saves directory)
        In case we don't have them tries to re-download them
        :return:
        """
        try:
            data = restore(os.path.join(self._saves_dir, "dataset"))
        except FileNotFoundError:
            data = self._fetch_data()
            write(os.path.join(self._saves_dir, "dataset"), data=data)
        return data

    @property
    def station_codes(self) -> Tuple[int, int]:
        """
        This method checks if we have already download the dataset for all station codes(stored as zip file
        in saves directory)
        In we case we don't it tries to re-download
        :return: a string of the file name
        """
        file_name = self.config['dataset_file_name']
        file_name_path = os.path.join(self._saves_dir, file_name)
        if not os.path.isfile(file_name_path):
            self._download_station_codes(dataset_url=self.config['dataset_url'],
                                         dataset_file_name=os.path.join(self._saves_dir, file_name))

        return self._get_station_codes(file_name_path)

    def _read_config(self) -> dict:
        """
        :return: the configuration as a dictionary
        :raises FileNotFoundError in case config file is not exists
        """
        config_path = os.path.join(self.current_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError('Config file is missing')

        with open(config_path) as config_file:
            config = json.load(config_file)
        return config

    def _make_directories(self, saves_folder_name: str, results_folder_name: str) -> None:
        """
        Creates the directory for saves and for the final results.
        """
        saves_dir = None
        results_dir = None
        try:
            saves_dir = os.path.join(self.parent_path, saves_folder_name)
            results_dir = os.path.join(self.parent_path, results_folder_name)
            os.mkdir(saves_dir)
            os.mkdir(results_dir)
        except FileExistsError:
            Logger.logger.debug(f"Folder name: {saves_folder_name} for saves already exists.")
            Logger.logger.debug(f"Folder name: {results_folder_name} for results already exists.")
        else:
            Logger.logger.debug(f"Successfully created the folder: {saves_folder_name}")
            Logger.logger.debug(f"Successfully created the folder: {results_folder_name}")
        finally:
            self._saves_dir = saves_dir
            self._result_dir = results_dir

    @logged
    def _fetch_data(self) -> List[dict]:
        """
        Fetch the requested data  from the public servers for the specific requested journey
        This query will return a list of travel times as well as benchmark travel times between
        an origin-destination (O-D) pair during the time period defined in the call.
        Travel times are flagged if they  are above certain
        A maximum time span of 7 days is allowed between from_datetime and to_datetime
        :return:
        """
        departure_code, arrival_code = self.station_codes
        start_time = datetime.time(*self.config['start_time'])
        end_time = datetime.time(*self.config['end_time'])
        start_date = datetime.date(*self.config['start_date'])
        end_date = datetime.date(*self.config['end_date'])
        params = self.config['request_params']
        # Gather data from the last three years
        i_days = 0
        trips_data = []
        Logger.logger.info(f"Starting pulling data until: {end_date.strftime('%d, %b %Y')}")
        while True:
            check_date = start_date + datetime.timedelta(days=i_days)
            Logger.logger.debug(f"Fetching data from date: {check_date.strftime('%d, %b %Y')}")
            if check_date > end_date:
                break
            from_time = datetime.datetime.combine(date=check_date, time=start_time)
            to_time = datetime.datetime.combine(date=check_date, time=end_time)
            from_datetime = str(int(from_time.timestamp()))
            to_datetime = str(int(to_time.timestamp()))
            params['from_datetime'] = from_datetime
            params['to_datetime'] = to_datetime
            params['api_key'] = self.config['api_key']
            params['from_stop'] = departure_code
            params['to_stop'] = arrival_code
            try:
                response = requests.get(url=self.config['base_url'],
                                        params=params,
                                        headers={"Accept": "application/json"})
                if response and response.status_code == 200:
                    as_json = response.json()["travel_times"]
                    for trip in as_json:
                        trips_data.append(
                            {"departure": datetime.datetime.fromtimestamp(float(trip["dep_dt"])),
                             "arrival": datetime.datetime.fromtimestamp(float(trip["arr_dt"])),
                             "travel_time_sec": trip['travel_time_sec']
                             }
                        )

                    Logger.logger.debug(f"Routes for: {check_date.strftime('%d, %b %Y')}: #{len(as_json)}")
                else:
                    Logger.logger.error(f"Bad request with code:{response.status_code}")
            except HTTPError as http_error:
                Logger.logger.error(http_error)

            i_days += 1
        return trips_data

    @logged
    def _get_station_codes(self, file_name: str) -> Tuple[int, int]:
        """
        This methods retrieves the location codes for the requested departure and arrival stations (declared in the
        config) with the help of the station code -> stops file
        :return:
        """
        with ZipFile(file_name) as zip_file:
            with zip_file.open('stops.txt') as stops_file:
                data = pd.read_csv(stops_file)
                df = pd.DataFrame(data,
                                  columns=['stop_id', 'stop_name', 'stop_desc', 'stop_url'])
                start = df.loc[df['stop_desc'] == self.config['departure']]
                departure_code = int(start['stop_id'].values)
                stop = df.loc[df['stop_desc'] == self.config['arrival']]
                arrival_code = int(stop['stop_id'].values)
                Logger.logger.info(f"Departure location code is:{departure_code}")
                Logger.logger.info(f"Arrival location code is:{arrival_code}")
                return departure_code, arrival_code

    @staticmethod
    def _download_station_codes(dataset_url: str, dataset_file_name: str) -> None:
        """
        Helper function for download the dataset.
        :return:
        """
        Logger.logger.debug("Downloading the data set for station codes...")
        try:
            response = requests.get(url=dataset_url, headers={'User-agent': 'Mozilla/5.0'})
            Logger.logger.debug(">Response status code is: {}".format(response.status_code))
            Logger.logger.debug(">Creating file ")
            with open(dataset_file_name, "wb") as writer:
                writer.write(response.content)
        except HTTPError as http_error:
            Logger.logger.error(http_error)


if __name__ == "__main__":
    os.environ["TZ"] = "US/Eastern"
