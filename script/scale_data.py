def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="Newspaper Layout Prediction")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default=DATA_PATH,
        help="Path for folder with images to be segmented. Images need to be png or jpg. Otherwise they"
             " will be skipped",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        dest="scale",
        default=1,
        help="Downscaling factor of the images. Polygon data will be upscaled accordingly",
    )