import math

from PIL import Image


def update_header_info(header: dict, bcorrect: int, pcorrect: int,
                       sinrhomax=0.997, rsun=6.96e8):
    """
    Recalculating header values using additional b and p angles and calculating
    additional values(sin and cos values of p and b angles, ) for projections
    """
    updated_header = dict()
    """
    Expected units for following values:
    x0 - pixels
    y0 - pixels
    p0 - degrees
    b0 - degrees
    r0 - pixels
    pixsize - radians
    """
    x0 = header['CRPIX1']
    y0 = header['CRPIX2']
    p0 = header['SOLAR_P']
    b0 = header['SOLAR_B0']
    r0 = header['IMAGE_R0']
    pixsize = header['CDELT1']

    p0 = p0 + pcorrect
    b0 = b0 + bcorrect
    s0 = r0 * pixsize
    pixsize = pixsize * math.pi / (180 * 3600)
    sr = r0 * pixsize
    rsun = 6.96e8
    dsun = rsun / math.sin(sr)
    radsol = r0 * math.cos(sr)
    rsqtest = (r0 ** 2) * (sinrhomax ** 2)
    sins0 = math.sin(sr)
    sinp0 = math.sin(math.radians(p0))
    cosp0 = math.cos(math.radians(p0))
    sinb0 = math.sin(math.radians(b0))
    cosb0 = math.cos(math.radians(b0))

    """
    Updating header with new set of values and corrected angles for b and p.
    """
    updated_header['x0'] = x0
    updated_header['y0'] = y0
    updated_header['p0'] = p0
    updated_header['s0'] = s0
    updated_header['b0'] = b0
    updated_header['r0'] = r0
    updated_header['pixsize'] = pixsize
    updated_header['dsun'] = dsun
    updated_header['radsol'] = radsol
    updated_header['sins0'] = sins0
    updated_header['sinp0'] = sinp0
    updated_header['cosp0'] = cosp0
    updated_header['sinb0'] = sinb0
    updated_header['cosb0'] = cosb0
    updated_header['rsqtest'] = rsqtest

    return updated_header


def pltobj2img(fig):
    """
    Converts matplotlib plot to PIL image object.
    :param fig: Matplotlib plot object.
    :return: image
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', transparent=True, pad_inches=0.0)
    buf.seek(0)
    return Image.open(buf)
