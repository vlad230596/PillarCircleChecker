import cv2
import glob
import numpy as np
import sys
from dataclasses import dataclass

from EniPy import colors
from EniPy import eniUtils

from typing import Final

MaxSampleValue: Final[int] = 2048

@dataclass
class SampleValue:
    native: int = 0

    def __init__(self, value):
        self.native = value

    def toRelative(self):
        return (self.native / MaxSampleValue)

    def toAbsRelative(self):
        return (self.toRelative() / 2 + 0.5)

    def toPixel(self, size) -> int:
        return int(self.toAbsRelative() * size)

class ResultView:
    def __init__(self, value, expected, negative_tolerance, positive_tolerance):
        self.value = value
        self.expected = expected
        self.negative_tolerance = negative_tolerance
        self.positive_tolerance = positive_tolerance

    def get_text_color(self):
        if self.negative_tolerance is not None and self.expected - self.value > self.negative_tolerance:
            return colors.Red
        if self.positive_tolerance is not None and self.value - self.expected > self.positive_tolerance:
            return colors.Red
        return colors.Green


class ResultExpectedView(ResultView):
    def __init__(self, value, expected, tolerance=0.0):
        ResultView.__init__(self, value, expected, tolerance, tolerance)

class ResultMinView(ResultView):
    def __init__(self, value, min_value):
        ResultView.__init__(self, value, min_value, negative_tolerance=0, positive_tolerance=None)


def process(path):
    reportsPath = glob.glob(f'{path}/*.json')
    for reportPath in reportsPath:
        print(f'\nProcessed: {reportPath}')
        size = 1024
        blank = np.zeros((size, size, 3), np.uint8)

        report = eniUtils.readJson(reportPath)
        sn = report['NodeProperties']['sys/HardwareSerialNumber']
        alsDataList = report['AlsData']
        contour = np.zeros((1, len(alsDataList), 2), np.float32)
        index = 0
        for alsData in alsDataList:
            sample = alsData['Sample']
            xSample = SampleValue(sample['x'])
            ySample = SampleValue(sample['y'])
            x = xSample.toPixel(size)
            y = ySample.toPixel(size)
            contour[0][index] = (xSample.toRelative(), -ySample.toRelative())
            blank[size - y, x] = colors.Green
            index = index + 1

        halfSize = int(size / 2)
        cv2.line(blank, (halfSize, 0), (halfSize, size), colors.Blue, 1)
        cv2.line(blank, (0, halfSize), (size, halfSize), colors.Blue, 1)

        (x, y), radius = cv2.minEnclosingCircle(contour)
        xPixel = int((x / 2 + 0.5) * size)
        yPixel = int((y / 2 + 0.5) * size)
        cv2.circle(blank, (xPixel, yPixel), int(radius / 2 * size), colors.Red)
        cv2.circle(blank, (xPixel, yPixel), 1, colors.Red, thickness=1)
        cv2.circle(blank, (xPixel, yPixel), int(size / 256), colors.Red, thickness=1)
        cv2.putText(blank, f'{reportPath}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Magenta, 1)
        cv2.putText(blank, f'{sn}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Magenta, 1)

        r_x = ResultExpectedView(x, 0.0, 0.05)
        r_y = ResultExpectedView(y, 0.0, 0.05)
        r_radius = ResultMinView(radius, 0.5)
        r_z = ResultExpectedView(alsDataList[0]["Sample"]["z"]/MaxSampleValue, 1.0, 0.01)

        cv2.putText(blank, f'x: {r_x.value:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_x.get_text_color(), 1)
        cv2.putText(blank, f'y: {r_y.value:.2f}', (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_y.get_text_color(), 1)

        cv2.putText(blank, f'radius: {r_radius.value:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_radius.get_text_color(), 1)
        cv2.putText(blank, f'z: {r_z.value:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, r_z.get_text_color(), 1)

        cv2.imshow('blank', blank)
        k = cv2.waitKey()
        if k == -1 or k == 27:
            break



if __name__ == '__main__':
    path = './reports/'
    if len(sys.argv) > 1:
        path = sys.argv[1]

    process(path)
