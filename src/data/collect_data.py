import json

import requests
import yaml

with open("configs/data/collect_config.yaml") as f:
    cfg = yaml.safe_load(f)

for source, source_dict in cfg["data_source"].items():
    url = source_dict["url"]

    params = {}
    params[source_dict["start_key"]] = cfg["datetime_range"]["start"]
    params[source_dict["end_key"]] = cfg["datetime_range"]["end"]

    if source == "weather":
        for station, station_coords in source_dict["station_coords"].items():
            print(f"Fetching data for station: {station}\n...")
            params["lat"] = station_coords[0]
            params["lon"] = station_coords[1]

            r = requests.get(url, params).json()
            with open(f"data/external/{station}_weather.json", "w") as f:
                json.dump(r, f)
            print(f"Saved data: {source}, {station}\n--------")

    else:
        print(f"Fetching data for source: {source}\n...")
        r = requests.get(url, params).json()
        with open(f"data/external/{source}.json", "w") as f:
            json.dump(r, f)
        print(f"Saved data: {source}\n--------")
