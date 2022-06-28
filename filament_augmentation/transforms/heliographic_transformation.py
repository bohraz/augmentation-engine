import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import math
import numpy as np
from projection_helper import update_header_info, pltobj2img


class Heliographic:
    """
     Function for loading 2 HMI magneto grams with a time gap loads parameters
     necessary for to heliographic
     bcorrect: B-angle correction in degrees to the solar rotation axis (angle
                in/out of plane of sky)
     pcorrect: P-angle (rotation in plane of sky) correction to the solar
                rotation axis in degrees

     Heliographic projections is calculated based on the following formula:
      Dave's formula based on B1900 epoch:
      meanEclipticLongitude = mod(279.696678+36000.768925*(jd - 2415020.0 + 0.5)/36525.,360.0);
      meanAnomaly = mod(358.475844 + 35999.049750*(jd - 2415020.0 + 0.5)/36525.,360.0)*pi/180.;
      eclipticLongitude = meanEclipticLongitude + (1.91947 - 0.00478*(jd - 2415020.0 + 0.5)/36525.)*sin(meanAnomaly) + 0.020*sin(2.*meanAnomaly);

    For more information on heliographic coordinates: https://www.aanda.org/articles/aa/full/2006/14/aa4262-05/aa4262-05.html
    """
    def __init__(self, pcorrect, bcorrect):
        """
        :param bcorrect: additional b angle for latitudinal translation, while keeping p angle 0 or original.
        :param pcorrect: additional p angle for rotational translation, while keeping b angle 0 or original.
        """
        self.pcorrect = pcorrect
        self.bcorrect = bcorrect
        self.ann_obj = dict()
        self.ilon = []
        self.ilat = []

    def __call__(self, data: list) -> torch.Tensor:
        """
        :param data: data is list of object containing image,
        annotation object and bounding box values.
        :return: Image tensor.
        """
        img, ann_obj, bbox = data
        self.ann_obj = update_header_info(ann_obj, self.pcorrect, self.bcorrect)
        self.__project(bbox)
        """
        Arranging latitude and longitude values based on image.
        """
        fig, ax = plt.subplots(1)
        ax.pcolor(self.ilon, self.ilat, img, cmap='gray')
        ax.axis('off')
        result_img = pltobj2img(plt)
        toTensor = T.ToTensor()
        return toTensor(result_img)

    def __project(self, bbox: list):
        px1, px2, px3, px4 = bbox
        """
        Initializing Latitude and longitude based on the size of filament.
        """
        self.ilat = np.empty((int(px4) + 1, int(px3)))
        self.ilat[:] = np.nan
        self.ilon = np.empty((int(px4) + 1, int(px3)))
        self.ilon[:] = np.nan
        """
        Get the values from header.
        """
        x0 = self.ann_obj['x0']
        y0 = self.ann_obj['y0']
        rsqtest = self.ann_obj['rsqtest']
        radsol = self.ann_obj['radsol']
        sins0 = self.ann_obj['sins0']
        cosp0 = self.ann_obj['cosp0']
        sinb0 = self.ann_obj['sinb0']
        cosb0 = self.ann_obj['cosb0']
        sinp0 = self.ann_obj['sinp0']

        for i in range(int(px4)):
            for j in range(int(px3)):
                x = px2 + i - x0
                y = px1 + j - y0
                rhoi2 = x ** 2 + y ** 2
                if rhoi2 < rsqtest:
                    rhoi = math.sqrt(rhoi2)
                    """
                    Solve quadratic equation to find zp
                    """
                    p = radsol / sins0
                    a = 1.00 + rhoi2 / (p * p)
                    b = -2.00 * rhoi2 / p
                    c = rhoi2 - radsol * radsol
                    b24ac = b * b - 4.00 * a * c
                    if b24ac > 0:
                        zp = (-b + math.sqrt(b24ac)) / (2 * a)
                        rhop = math.sqrt(radsol * radsol - zp * zp)
                        if (x * y) != 0:
                            xp = rhop * x / rhoi
                            yp = xp * y / x
                        elif x == 0:
                            xp = 0
                            yp = np.sign(y) * rhop
                        elif y == 0:
                            yp = 0
                            xp = np.sign(x) * rhop
                        """
                        Rotated Heliographic cartesian coordinates with new P-angle
                        """
                        xb = xp * cosp0 + yp * sinp0
                        yb = -xp * sinp0 + yp * cosp0
                        zb = zp
                        """
                        Rotated Heliographic cartesian coordinates with B-angle
                        """
                        xs = xb
                        ys = yb * cosb0 + zb * sinb0
                        """
                        Converting the following calculated values to latitude and longitude.
                        """
                        self.ilat[i, j] = math.asin(ys / radsol)
                        self.ilon[i, j] = math.asin(xs / (radsol * math.cos(self.ilat[i, j])))


if __name__ == "__main__":
    img = Image.open(
        r'D:\GSU_Assignments\Semester_2\DL\augmentation_engine_backup\evalutate_augmentation_engine\filament_images\L\2015083118183905.jpg')
    flip = T.Compose([Heliographic(30, 45), T.ToPILImage()])
    header ={
            "CRPIX1": 1024,
            "CRPIX2": 1025,
            "SOLAR_P": 0.0,
            "SOLAR_B0": 7.17546,
            "IMAGE_R0": 951,
            "CDELT1": 1.0015,
            "CDELT2": 1.0015
        }
    bbox = [
                565.034947578632,
                1259.8227658512233,
                177.12730903644547,
                195.72041937094355
            ]
    data = [img ,header, bbox]
    flip_img = flip(data)
    plt.imshow(flip_img, cmap="gray")
    plt.show()
