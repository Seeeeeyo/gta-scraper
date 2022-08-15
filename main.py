import argparse
import time

import pyscreenshot as ImageGrab
import schedule

import fiftyone as fo

from datetime import datetime

ROAD_CONDITION_CATEGORIES = ["dry", "wet", "snowy"]

class GtaScreenshotScraper():
    def __init__(self, dataset_name=None):
        assert isinstance(dataset_name, str)
        self.dataset_name = dataset_name
        self.dataset = fo.load_dataset(dataset_name)
        self.dataset.persistent = True

    def create_sample_from_context(
            self,
            image_name,
            sequence_tag=None,
            road_condition=None,
    ):
        sample = fo.Sample(filepath=image_name)
        if sequence_tag is not None:
            sample["tags"].append(sequence_tag)
        if road_condition is not None:
            assert road_condition in ROAD_CONDITION_CATEGORIES
            sample["gt_road_condition"] = fo.Classifications(classifications=[fo.Classification(label=road_condition)])
        return sample

    def take_screenshot(self):
        image_name = f"screenshot-{str(datetime.now())}"
        screenshot = ImageGrab.grab()
        width, height = screenshot.size
        cropped_img = screenshot.crop((width / 6, height / 6, 5 * width / 6, 5 * height / 6))  # left, top, right, bottom
        filepathloc = f"screenshots/{image_name}.png"
        cropped_img.save(filepathloc)
        print("Screenshot taken.")

        sample = self.create_sample_from_context(
            filepathloc,
            sequence_tag='test',
            road_condition='dry',
        )

        self.dataset.add_sample(sample)
        sample.save()
        print("Sample saved.")


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='esmart_gta', type=str,
                        help="The dataset name to store the samples in (51).")
    parser.add_argument("--interval", default=5, type=int,
                        help="Time interval (seconds) between the screenshots.")
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    gta = GtaScreenshotScraper(dataset_name=opt.dataset)
    schedule.every(opt.interval).seconds.do(gta.take_screenshot)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
