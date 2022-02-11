__author__ = "Shreejaa Talla"

import numpy as np
import math


def latlon2pixel(lon, lat, image, header):
    # image = np.array(image)
    size = np.array(image).shape[0]
    vmap = np.empty((size, size))
    for i in range(len(lon)):
        for j in range(len(lon[0])):
            theta = lat[i][j]
            phi = lon[i][j]

            sintheta = math.sin(theta)
            costheta = math.cos(theta)
            sinphi = math.sin(phi)
            cosphi = math.cos(phi)

            radsol = header['radsol']
            xs = radsol * sinphi * costheta
            ys = radsol * sintheta
            zs = radsol * cosphi * costheta

            xb = xs
            yb = header['cosb0'] * ys - header['sinb0'] * zs
            zb = header['cosb0'] * zs + header['sinb0'] * ys

            xp = header['cosp0'] * xb - header['sinp0'] * yb
            yp = header['cosp0'] * yb + header['sinp0'] * xb
            zp = zb
            rhoi = 1
            zpmin = header['radsol'] * header['sins0']
            if zp > zpmin:
                rhop = math.sqrt(xp ** 2 + yp ** 2)
                thetap = 0
                if rhop > 0.1:
                    thetap = np.arctan2(yp, xp)
                rhoi = rhoi * (1.0 + zp / (header['radsol'] / header['sins0'] - zp))
                x = rhoi * math.cos(thetap)
                y = rhoi * math.sin(thetap)
                lambda_ = math.asin(rhop / header['radsol'])
                s = math.atan(rhoi * header['sins0'] / header['radsol'])

                xi = header['x0'] + x
                yi = header['y0'] + y

                radsq = (xi - header['x0']) * (xi - header['x0']) + (yi - header['y0']) * (yi - header['y0'])

                if radsq < header['rsqtest']:
                    ix = math.floor(xi)
                    iy = math.floor(yi)
                    dx = xi - ix
                    dy = yi - iy
                    data = image[ix - 1:ix + 3, iy - 1:iy + 3]
                    dx2 = dx ** 2
                    dx3 = dx ** 3
                    dy2 = dy ** 2
                    dy3 = dy ** 3
                    wx = np.zeros((4, 4))
                    wx1 = -0.50 * dx + dx2 - 0.50 * dx3
                    wx2 = 1.0 - 2.50 * dx2 + 1.50 * dx3
                    wx3 = 0.50 * dx + 2.0 * dx2 - 1.50 * dx3
                    wx4 = -0.50 * dx2 + 0.50 * dx3
                    # print(wx1,wx2,wx3,wx4)
                    wx[0:4, 0] = [wx1, wx2, wx3, wx4]
                    wx[0:4, 1] = wx[0:4, 0]
                    wx[0:4, 2] = wx[0:4, 0]
                    wx[0:4, 3] = wx[0:4, 0]

                    wy = np.zeros((4, 4))
                    wy[0, 0:4] = [(-0.50 * dy + dy2 - 0.50 * dy3), (1.0 - 2.50 * dy2 + 1.50 * dy3),
                                  (0.50 * dy + 2.0 * dy2 - 1.50 * dy3), (-0.50 * dy2 + 0.50 * dy3)]
                    wy[1, 0:4] = wy[0, 0:4]
                    wy[2, 0:4] = wy[0, 0:4]
                    wy[3, 0:4] = wy[0, 0:4]

                    weight = wx * wy
                    # sum_w = sum(sum(weight*data))
                    vmap[i, j] = sum(sum(weight*data)) / 4
                else:
                    vmap[i, j] = np.nan

            else:
                vmap[i, j] = np.nan

    return vmap

if __name__ == "__main__":
    image_path = r'D:\GSU_Assignments\Semester_2\DL\augmentation_engine\bbso_halph_fr_20120312_190913.jpg'
    # image_path = r'D:\GSU_Assignments\Semester_2\DL\evalutate_augmentation_engine\filament_images\L\2015083118183905.jpg'

    header = dict()
    header['CRPIX1'] = 1025
    header['CRPIX2'] = 1022
    header['SOLAR_P'] = 0.0
    header['SOLAR_B0'] = -7.20494
    header['IMAGE_R0'] = 965
    header['CDELT1'] = 1.00150
    header['CDELT2'] = 1.00150

    # header['CRPIX1'] = 1024
    # header['CRPIX2'] = 1025
    # header['SOLAR_P'] = 0.0
    # header['SOLAR_B0'] = 7.17546
    # header['IMAGE_R0'] = 951
    # header['CDELT1'] = 1.00150
    # header['CDELT2'] = 1.00150

    img = Image.open(image_path)
    # img = cv2.resize(np.array(img), (2048, 2048))
    # img = Image.fromarray(img)
    # image = img.transpose(Image.ROTATE_270)
    img, ilat, ilon, header = pixel2latlon(img, header, 0, 0)
    # plt.pcolor(ilon, ilat, img, cmap='gray')
    # plt.show()
    R = 6.96
    xmin = - math.pi * R
    xmax = math.pi * R
    ymin = R * math.log1p(math.tan(math.pi / 4 - math.pi / 4))
    ymax = R * math.log1p(math.tan(math.pi / 4 + math.pi / 4))
    x = np.linspace(xmin, xmax, 4096)
    dx = x[1] - x[0]
    dy = 1.3 * dx
    x = np.arange(-1024 * dx + dx / 2, 1024 * dx - dx / 2 + dx, dx)
    y = np.arange(-1024 * dy + dy / 2, 1024 * dy - dy / 2 + dy, dy)
    # x = -256*dx+dx/2:dx:256*dx-dx/2
    # y = -256*dy+dy/2:dy:256*dy-dy/2
    [x, y] = np.meshgrid(x, y)
    lon = x / R
    lat = 2 * np.arctan(np.exp(y / R)) - math.pi / 2
    # print(lon, lat)
    img = np.array(img, dtype='f')
    vmap = latlon2pixel(x, y, img, header)
    print(vmap)
