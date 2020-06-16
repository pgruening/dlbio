import collections
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as plt_patches

from DLBio.helpers import safe_division


class IRectangle(object):
    """Abstract base class for a rectagle. Intended to specify the positions
    of each dimension to reduce bugs.

    """
    x_pos = 0
    y_pos = 1
    w_pos = 2
    h_pos = 3

    def __init__(self, **kwargs):
        """
            Parameters
            ----------
            confidence : float in [0, 1]
                the value defines how certain this rectangle contains an object.
        """
        self.id = kwargs.pop('id', None)
        self.confidence = kwargs.pop('confidence', 1.0)

        if len(kwargs.keys()) > 0:
            warnings.warn(
                'Found ununsed keys: {}'.format(kwargs.keys())
            )

    def __repr__(self):
        return '{}: {}'.format(self.__class__.__name__, self.__dict__)

    def add_id(self, id):
        self.id = id


class TopLeftRectangle(IRectangle):
    """Rectangle containing pixel coordinates. x and y determine the top-left
    point of the rectangle. 
    """

    def __init__(self, **kwargs):
        self.x = kwargs.pop('x')
        self.y = kwargs.pop('y')
        self.h = kwargs.pop('h')
        self.w = kwargs.pop('w')
        super(TopLeftRectangle, self).__init__(**kwargs)

    def estimate_jaccard_index(self, rectangle):
        """Compute the intersection of the object and another TopLeftRectangle
        and divide it by the union.

        Parameters
        ----------
        rectangle : TopLeftRectangle

        Returns
        -------
        jaccard_index : float
        """
        # estimate position of intersection rectangle
        left = max(self.x, rectangle.x)
        top = max(self.y, rectangle.y)
        right = min(self.x + self.w, rectangle.x + rectangle.w)
        bottom = min(self.y + self.h, rectangle.y + rectangle.h)

        a = max(right - left, 0)
        b = max(bottom - top, 0)

        # estimate areas
        area_intersection = a * b
        area_union = self.w * self.h + rectangle.w * rectangle.h - area_intersection

        jaccard_index = safe_division(area_intersection, area_union)
        return jaccard_index

    def get_array(self):
        output = np.zeros(4)
        output[self.x_pos] = self.x
        output[self.y_pos] = self.y
        output[self.w_pos] = self.w
        output[self.h_pos] = self.h
        return output

    def point_is_within_rectangle(self, x, y):
        return x >= self.x and x <= self.x + self.w \
            and y >= self.y and y <= self.y + self.h

    def get_viewable(self, color=[255, 0, 0], type_id='unknown'):
        """Return a ViewableTopLeftRectangle with the current parameters.

        Parameters
        ----------
        color : list, optional
            in which color shall the rectangle be displayed 
            (the default is [255, 0, 0], which is blue)
        type_id : str, optional
            string specifying the type of rectangle (e.g. positive, negative, etc.).
            The default is 'unknown'.

        Returns
        -------
        ViewableTopLeftRectangle
        """
        input = self.__dict__
        input.update({'color': color, 'type_id': type_id})
        return ViewableTopLeftRectangle(**input)

    def to_c_rectangle(self, h_norm, w_norm):
        """Turn the rectangle into a CenterNormalizedRectangle.
        Pass the image size as well to know how the rectangle needs to
        be scaled.

        Parameters
        ----------
        h_norm : float
            scaling factor
        w_norm : float
            scaling factor

        Returns
        -------
        CenterNormalizedRectangle
        """
        if h_norm is None:
            h_norm = float(self.h)
        else:
            if type(h_norm) != float:
                raise ValueError('h_norm is not a float number')
        if w_norm is None:
            w_norm = float(self.w)
        else:
            if type(w_norm) != float:
                raise ValueError('w_norm is not a float number')

        cx = self.x + .5 * self.w
        cy = self.y + .5 * self.h
        return CenterNormalizedRectangle(
            cx=cx,
            cy=cy,
            w=self.w / h_norm,
            h=self.h / w_norm,
        )


class ViewableTopLeftRectangle(TopLeftRectangle):
    def __init__(self, **kwargs):
        self.color = kwargs.pop('color', [255, 0, 0])
        self.type_id = kwargs.pop('type_id', 'unknown')
        super(ViewableTopLeftRectangle, self).__init__(**kwargs)

    def add_cv_rectangle(self, image, color=None):
        """Draw a rectangle on the image via openCV

        Parameters
        ----------
        image : np.array
        """
        if color is None:
            color = self.color

        cv2.rectangle(
            image,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            color
        )

    def get_pyplot_patch(self):
        """Return a pyplot platch that can by added to a plot axis.

        Returns
        -------
        plt_patches
        """
        xy = [self.x, self.y]
        h_box = self.h
        w_box = self.w
        return plt_patches.Rectangle(
            xy, w_box, h_box,
            linewidth=1,
            edgecolor=np.array(self.color).astype('float') / 255.0,
            facecolor='none')


class CenterNormalizedRectangle(IRectangle):
    """Rectangle containing normalized positions (the typical output of a 
    single shot detector network). Furthermore, cx and cy define the center
    of the rectangle. 

    """

    def __init__(self, **kwargs):
        self.cx = kwargs.pop('cx')
        self.cy = kwargs.pop('cy')
        self.h = kwargs.pop('h')
        self.w = kwargs.pop('w')
        super(CenterNormalizedRectangle, self).__init__(**kwargs)

    def get_normalized_top_left(self):
        # function is needed only for ssd-label computations
        x = self.cx - .5 * self.w
        y = self.cy - .5 * self.h

        return TopLeftRectangle(
            x=x,
            y=y,
            w=self.w,
            h=self.h
        )

    def to_rectangle(self, h_norm, w_norm):
        """Returns a topleft rectangle

        Parameters
        ----------
        h_norm, w_norm : float
            The dimensions are multiplied by the current position values
            to compute the pixel positions. Be careful to choose the right
            value here!

        Returns
        -------
        TopLeftRectangle
        """
        # get top left normalized coordinate
        x = self.cx - .5 * self.w
        y = self.cy - .5 * self.h

        # compute float normalized values to pixel coordinates
        x_new = int(np.round(x * w_norm))
        y_new = int(np.round(y * h_norm))
        h_new = int(np.round(self.h * h_norm))
        w_new = int(np.round(self.w * w_norm))

        return TopLeftRectangle(
            x=x_new,
            y=y_new,
            w=w_new,
            h=h_new,
        )

    def estimate_jaccard_index(self, c_rectangle, h_norm, w_norm):
        # special case for ssd label computations
        if h_norm == 1.0 and w_norm == 1.0:
            rect1 = self.get_normalized_top_left()
            rect2 = c_rectangle.get_normalized_top_left()
        else:
            rect1 = self.to_rectangle(h_norm, w_norm)
            rect2 = c_rectangle.to_rectangle(h_norm, w_norm)
        return rect1.estimate_jaccard_index(rect2)
