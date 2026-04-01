
import os
import json
import yaml

from omegaconf import DictConfig, OmegaConf



# Mapping from FGVC-Aircraft variant codes to descriptive names for CLIP
AIRCRAFT_DESCRIPTIVE_NAMES = {
    "707-320": "Boeing 707-320 four-engine jet airliner",
    "727-200": "Boeing 727-200 three-engine jet airliner",
    "737-200": "Boeing 737-200 narrow-body twin-engine jet",
    "737-300": "Boeing 737-300 narrow-body twin-engine jet",
    "737-400": "Boeing 737-400 narrow-body twin-engine jet",
    "737-500": "Boeing 737-500 narrow-body twin-engine jet",
    "737-600": "Boeing 737-600 narrow-body twin-engine jet",
    "737-700": "Boeing 737-700 narrow-body twin-engine jet",
    "737-800": "Boeing 737-800 narrow-body twin-engine jet",
    "737-900": "Boeing 737-900 narrow-body twin-engine jet",
    "747-100": "Boeing 747-100 wide-body four-engine jumbo jet",
    "747-200": "Boeing 747-200 wide-body four-engine jumbo jet",
    "747-300": "Boeing 747-300 wide-body four-engine jumbo jet",
    "747-400": "Boeing 747-400 wide-body four-engine jumbo jet",
    "757-200": "Boeing 757-200 narrow-body twin-engine jet",
    "757-300": "Boeing 757-300 narrow-body twin-engine jet",
    "767-200": "Boeing 767-200 wide-body twin-engine jet",
    "767-300": "Boeing 767-300 wide-body twin-engine jet",
    "767-400": "Boeing 767-400 wide-body twin-engine jet",
    "777-200": "Boeing 777-200 wide-body twin-engine jet",
    "777-300": "Boeing 777-300 wide-body twin-engine jet",
    "A300B4": "Airbus A300B4 wide-body twin-engine jet",
    "A310": "Airbus A310 wide-body twin-engine jet",
    "A318": "Airbus A318 narrow-body twin-engine jet",
    "A319": "Airbus A319 narrow-body twin-engine jet",
    "A320": "Airbus A320 narrow-body twin-engine jet",
    "A321": "Airbus A321 narrow-body twin-engine jet",
    "A330-200": "Airbus A330-200 wide-body twin-engine jet",
    "A330-300": "Airbus A330-300 wide-body twin-engine jet",
    "A340-200": "Airbus A340-200 wide-body four-engine jet",
    "A340-300": "Airbus A340-300 wide-body four-engine jet",
    "A340-500": "Airbus A340-500 wide-body four-engine jet",
    "A340-600": "Airbus A340-600 wide-body four-engine jet",
    "A380": "Airbus A380 double-deck wide-body four-engine jet",
    "ATR-42": "ATR 42 twin-turboprop regional airliner",
    "ATR-72": "ATR 72 twin-turboprop regional airliner",
    "An-12": "Antonov An-12 four-engine turboprop transport",
    "BAE-125": "British Aerospace BAe 125 twin-engine business jet",
    "BAE_146-200": "British Aerospace BAe 146-200 four-engine regional jet",
    "BAE_146-300": "British Aerospace BAe 146-300 four-engine regional jet",
    "Beechcraft_1900": "Beechcraft 1900 twin-turboprop commuter",
    "Boeing_717": "Boeing 717 narrow-body twin-engine jet",
    "C-130": "Lockheed C-130 Hercules four-engine turboprop military transport",
    "C-47": "Douglas C-47 Skytrain twin piston-engine transport",
    "CRJ-200": "Bombardier CRJ-200 twin-engine regional jet",
    "CRJ-700": "Bombardier CRJ-700 twin-engine regional jet",
    "CRJ-900": "Bombardier CRJ-900 twin-engine regional jet",
    "Cessna_172": "Cessna 172 Skyhawk single-engine propeller light aircraft",
    "Cessna_208": "Cessna 208 Caravan single-engine turboprop utility",
    "Cessna_525": "Cessna Citation CJ series twin-engine light business jet",
    "Cessna_560": "Cessna Citation V twin-engine business jet",
    "Challenger_600": "Bombardier Challenger 600 twin-engine business jet",
    "DC-10": "McDonnell Douglas DC-10 wide-body three-engine jet",
    "DC-3": "Douglas DC-3 twin piston-engine propeller airliner",
    "DC-6": "Douglas DC-6 four piston-engine propeller airliner",
    "DC-8": "Douglas DC-8 four-engine jet airliner",
    "DC-9-30": "McDonnell Douglas DC-9-30 narrow-body twin-engine jet",
    "DH-82": "de Havilland DH.82 Tiger Moth biplane trainer",
    "DHC-1": "de Havilland Canada DHC-1 Chipmunk single-engine trainer",
    "DHC-6": "de Havilland Canada DHC-6 Twin Otter twin-turboprop",
    "DHC-8-100": "de Havilland Canada Dash 8-100 twin-turboprop regional",
    "DHC-8-300": "de Havilland Canada Dash 8-300 twin-turboprop regional",
    "DR-400": "Robin DR 400 single-engine light aircraft",
    "Dornier_328": "Dornier 328 twin-turboprop regional airliner",
    "E-170": "Embraer E-170 twin-engine regional jet",
    "E-190": "Embraer E-190 twin-engine regional jet",
    "E-195": "Embraer E-195 twin-engine regional jet",
    "EMB-120": "Embraer EMB 120 Brasilia twin-turboprop",
    "ERJ_135": "Embraer ERJ 135 twin-engine regional jet",
    "ERJ_145": "Embraer ERJ 145 twin-engine regional jet",
    "Embraer_Legacy_600": "Embraer Legacy 600 twin-engine business jet",
    "Eurofighter_Typhoon": "Eurofighter Typhoon twin-engine delta-wing fighter jet",
    "F-16A-B": "General Dynamics F-16 Fighting Falcon single-engine fighter jet",
    "F-A-18": "McDonnell Douglas F/A-18 Hornet twin-engine fighter jet",
    "Falcon_2000": "Dassault Falcon 2000 twin-engine business jet",
    "Falcon_900": "Dassault Falcon 900 three-engine business jet",
    "Fokker_100": "Fokker 100 twin-engine regional jet",
    "Fokker_50": "Fokker 50 twin-turboprop regional airliner",
    "Fokker_70": "Fokker 70 twin-engine regional jet",
    "Global_Express": "Bombardier Global Express twin-engine long-range business jet",
    "Gulfstream_IV": "Gulfstream IV twin-engine business jet",
    "Gulfstream_V": "Gulfstream V twin-engine long-range business jet",
    "Hawk_T1": "BAE Systems Hawk T1 single-engine jet trainer",
    "Il-76": "Ilyushin Il-76 four-engine jet transport",
    "L-1011": "Lockheed L-1011 TriStar wide-body three-engine jet",
    "MD-11": "McDonnell Douglas MD-11 wide-body three-engine jet",
    "MD-80": "McDonnell Douglas MD-80 narrow-body twin-engine jet",
    "MD-87": "McDonnell Douglas MD-87 narrow-body twin-engine jet",
    "MD-90": "McDonnell Douglas MD-90 narrow-body twin-engine jet",
    "Metroliner": "Fairchild Swearingen Metroliner twin-turboprop commuter",
    "Model_B200": "Beechcraft King Air B200 twin-turboprop",
    "PA-28": "Piper PA-28 Cherokee single-engine light aircraft",
    "SR-20": "Cirrus SR-20 single-engine composite light aircraft",
    "Saab_2000": "Saab 2000 twin-turboprop regional airliner",
    "Saab_340": "Saab 340 twin-turboprop regional airliner",
    "Spitfire": "Supermarine Spitfire single-engine fighter with elliptical wing",
    "Tornado": "Panavia Tornado twin-engine variable-sweep wing fighter",
    "Tu-134": "Tupolev Tu-134 twin-engine rear-mounted jet airliner",
    "Tu-154": "Tupolev Tu-154 three-engine jet airliner",
    "Yak-42": "Yakovlev Yak-42 three-engine jet airliner",
}


def get_aircraft_descriptive_name(code):
    """Map aircraft variant code to a descriptive name for CLIP."""
    return AIRCRAFT_DESCRIPTIVE_NAMES.get(code, code)



def get_class_order(file_name: str) -> list:
    r"""TO BE DOCUMENTED"""
    with open(file_name, "r+") as f:
        data = yaml.safe_load(f)
        return data["class_order"]


def get_class_ids_per_task(args):
    yield args.class_order[:args.initial_increment]
    for i in range(args.initial_increment, len(args.class_order), args.increment):
        yield args.class_order[i:i + args.increment]

def get_class_names(classes_names, class_ids_per_task):
    return [classes_names[class_id] for class_id in class_ids_per_task]


def get_dataset_class_names(workdir, dataset_name, long=False):
    with open(os.path.join(workdir, "dataset_reqs", f"{dataset_name}_classes.txt"), "r") as f:
        lines = f.read().splitlines()
    return [line.split("\t")[-1] for line in lines]


def save_config(config: DictConfig) -> None:
    OmegaConf.save(config, "config.yaml")


def get_workdir(path):
    split_path = path.split("/")
    workdir_idx = split_path.index("RAPF") # If a 'ValueError' occurs, replace 'RAPF' with your actual work directory
    return "/".join(split_path[:workdir_idx+1])


