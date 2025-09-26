from . import koiprediction


def run_prediction(file_path: str, satellite: str):
    """
    Central engine to route prediction requests
    to the correct satellite-specific predictor.
    """
    satellite = satellite.upper().strip()

    if satellite == "KOI":
        return koiprediction.predict_from_file(file_path)
    elif satellite == "K2":
        raise NotImplementedError("K2 prediction not yet implemented.")
    elif satellite == "TOI":
        raise NotImplementedError("TOI prediction not yet implemented.")
    else:
        raise ValueError(f"Unknown satellite: {satellite}")
