from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor

import numpy


def patch_as_scalar(a):
    return a.item()


setattr(numpy, "asscalar", patch_as_scalar)


def rgb2lab(color_input: tuple[int, int, int]) -> list[int]:
    num = 0
    RGB = [0, 0, 0]
    for value in color_input:
        value = float(value) / 255
        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92
        RGB[num] = value * 100
        num = num + 1
    XYZ = [0, 0, 0, ]
    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)
    XYZ[0] = float(XYZ[0]) / 95.047  # ref_X =  95.047   Observer= 2°, Illuminant= D65
    XYZ[1] = float(XYZ[1]) / 100.0  # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883  # ref_Z = 108.883
    num = 0
    for value in XYZ:
        if value > 0.008856:
            value = value ** 0.3333333333333333
        else:
            value = (7.787 * value) + (16 / 116)
        XYZ[num] = value
        num = num + 1
    Lab = [0, 0, 0]
    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])
    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)
    return Lab


def delta_e(rgb1: tuple[int, int, int], rgb2: tuple[int, int, int]) -> float:
    lab1 = rgb2lab(rgb1)
    lab2 = rgb2lab(rgb2)
    # CIE 2000 ΔE 계산
    return delta_e_cie2000(LabColor(lab1[0], lab1[1], lab1[2]), LabColor(lab2[0], lab2[1], lab2[2]))


if __name__ == "__main__":
    # 예시 색상
    color_a = (3, 35, 61)
    color_b = (32, 40, 53)

    delta_e_value = delta_e(color_a, color_b)
    print("두 색상 간의 ΔE:", delta_e_value)
