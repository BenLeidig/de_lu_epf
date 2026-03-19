import json
from pathlib import Path

import requests
import yaml

if __name__ == "__main__":
    print("+" * 8, " `collect_data.py` started. ", "+" * 8)

    # Set paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    config_path = BASE_DIR / "configs"
    data_path = BASE_DIR / "data/external"

    # Set configs
    with open(config_path / "data/collect_config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Fetching data for each source given the config params
    for source, source_dict in cfg["data_source"].items():
        url = source_dict["url"]

        ## Set main params for each query
        params = {}
        params[source_dict["start_key"]] = cfg["datetime_range"]["start"]
        params[source_dict["end_key"]] = cfg["datetime_range"]["end"]

        ## Weather source needs to be looped to get data across all specified stations
        ### (will average this later in src/data/processing/process_data.py across stations)
        if source == "weather":
            for station, station_coords in source_dict["station_coords"].items():
                print(
                    "-" * 8, f"Fetching data for source: {station} weather...", "-" * 8
                )
                params["lat"] = station_coords[0]
                params["lon"] = station_coords[1]

                r = requests.get(url, params).json()
                with open(data_path / f"{station}_weather.json", "w") as f:
                    json.dump(r, f)
                print(f" Saved data for source: {station} weather.")

        ## No other source needs to be looped
        ### (i.e. single query for each 'other' source)
        else:
            print("-" * 8, f"Fetching data for source: {source}...", "-" * 8)
            r = requests.get(url, params).json()
            with open(data_path / f"{source}.json", "w") as f:
                json.dump(r, f)
            print(f"Saved data for source: {source}.")
    print("+" * 8, " `collect_data.py` completed. ", "+" * 8)
