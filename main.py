import cv2
import glob
import numpy as np
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
        cv2.putText(blank, f'x: {x:.2f} y: {y:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Magenta, 1)
        cv2.putText(blank, f'radius: {radius:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.Magenta, 1)

        cv2.imshow('blank', blank)
        cv2.waitKey()


if __name__ == '__main__':
    process('./reports/')
