import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math
import numpy as np
from projection_helper import update_header_info, pltobj2img
from time import perf_counter
from torchvision.transforms.functional import resize


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

        # Convert img to numpy array
        img_array = np.array(img)

        px1, px2 = bbox[0], bbox[1]  # Top-left point
        px3, px4 = bbox[2], bbox[3]  # Width and height

        # Round bbox floats to integers for slicing and create image cutout(y comes first in numpy)
        x1, y1 = int(round(px1)), int(round(px2))
        x2, y2 = int(round(px1 + px3)), int(round(px2 + px4))

        shading = 'flat' # or 'nearest'
        if shading == 'flat':
            x1 += 1
            y1 += 1

        cutout = img_array[y1:y2, x1:x2]

        #Draw rectangle around cutout area
        shape = [(x1,y1), (x2,y2)]

        img1 = ImageDraw.Draw(img)
        img1.rectangle(shape, width=10, fill=None, outline="white")
        img.show()

        # for the bcorrect above 91 error, causes 3d transformation?
        # self.ilat = np.sort(self.ilat, axis=0)
        # self.ilon = np.sort(self.ilon, axis=1)

        ax.pcolormesh(self.ilon, self.ilat, cutout, cmap='gray', shading=shading)

        #ax.set_title("Heliographic Transformation Visualization")
        ax.axis('off')

        result_img = pltobj2img(plt)
        toTensor = T.ToTensor()
        return toTensor(result_img)

    def __project(self, bbox: list):
        px1, px2, px3, px4 = bbox
        """
        Initializing Latitude and longitude based on the size of filament.
        """
        self.ilat = np.empty((int(px4), int(px3)))
        self.ilat[:] = np.nan
        self.ilon = np.empty((int(px4), int(px3)))
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

## sample example
## get the following image data from "https://bitbucket.org/gsudmlab/bbso_data/downloads/" in 2015.zip file.
if __name__ == "__main__":
    img = Image.open(r'bbso_halph_fr_20150807_192111.jpg')#.convert('L') #2015083118183??
    flip = T.Compose([Heliographic(30, 30), T.ToPILImage()])
    header = {
        "CRPIX1": 1026,
        "CRPIX2": 1026,
        "SOLAR_P": 0.0,
        "SOLAR_B0": 6.19041,
        "IMAGE_R0": 946,
        "CDELT1": 1.0015,
        "CDELT2": 1.0015
    }
    # bbox = [
    #     1252.961907139291,
    #     1308.3809785322017,
    #     70.13409885172246,
    #     49.678382426360486
    # ]
    bbox = [
        303.22865701447836,
        1070.7044433349974,
        105.20119820269599,
        237.6764852720919
    ]

    start = perf_counter()
    data = [img, header, bbox]
    flip_img = flip(data)
    end = perf_counter()
    print("BENCHMARK: ", end - start)

    plt.imshow(flip_img, cmap="gray")
    plt.show()