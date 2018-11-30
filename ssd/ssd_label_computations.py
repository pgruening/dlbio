"""
Functions that generate a single-shot-detector label from an instance segmentation
image (cc_image).
"""

import collections

import cv2
import numpy as np

import DLBio.ssd.ssd_offset_calculator as oc
# from DLBio.helpers import(Rectangle, compute_connected_component_stats,
#                          cRectangle)
from DLBio.helpers import compute_connected_component_stats
from DLBio.ssd.rectangles import CenterNormalizedRectangle, TopLeftRectangle

NUM_BOXES = 4
S_MIN = 1e-5
S_MAX = .02
RATIOS = [.33, .5, 1, 2, 3]

# shape is a tuple (h, w) referring to the output's feature map shape.
# box_specs is a list of BoxSpec
Prior = collections.namedtuple('Prior', 'shape box_specs')

# Defines a generic AnchorBox by its index (to find it in the output feature-map)
# and its height and width.
BoxSpec = collections.namedtuple('BoxSpec', 'index h w')


def get_ssd_labels_fast(label, priors):
    """Return a SSD label vector of the connected-component image label.
    For each shape, the order is conf_0, conf_1, ..., conf_n, loc_0, ..., loc_n
    Shapes are then concatenated. Hence,
    s0_c0, ... , s0_cn, s0_l0, ..., s0_ln, s1_c0, ..., s1_cm, s1_l0, ..., 
    s1_lm, s2_c0, ...

    Each confidence map has an added background class. Hence, num_fmaps for
    conf_i is num_classes + 1

    Parameters
    ----------
    label : cc-image
        each int value determines to which instance the pixel belongs
    priors : list of Priors
        A prior is a tuple ((h,w), (box_spec0, box_spec1,...))
        For each prior's boxSpec, a feature map is created with size (h,w)
        That contains the confidence scores and offsets of the specific box.

    Returns
    -------
    np.array of shape (n,)
        Concatenated output for SSD network.
    """
    # -------------function definitions --------------------
    def setup_outputs_and_priors():
        """For each prior's box, create two  np.arrays with zeros of size
        (h,w, num_boxes )and (h,w,4*num_boxes) for confidence and location
        respectively.

        Returns
        -------
        list of (conf_array, loc_array, prior)
        """
        outputs_and_priors = []
        for prior in priors:
            num_boxes = len(prior.box_specs)
            h, w = prior.shape
            output_conf = np.zeros((h, w, num_boxes))
            output_loc = np.zeros((h, w, 4*num_boxes))
            outputs_and_priors.append([output_conf, output_loc, prior])
        if len(outputs_and_priors) == 0:
            raise ValueError('No priors found! Priors: {}'.format(priors))
        return outputs_and_priors

    def get_valid_area():
        """Find the extrem points of a set of rectangles.
        rectangles is a list of four rectangles that have a JI of .5 to the
        current ground truth box. The min and max values define the array in 
        which any other rectangle has a JI >= .5
        Returns
        -------
        4 floats
            x and y coordinates of the area where each rectangle with the 
            current shape has a JI >= .5 to the current ground truth box.
        """
        # (x,y,h,w)
        left = np.min(np.array([b.cx for b in rectangles]))
        right = np.max(np.array([b.cx for b in rectangles]))
        top = np.min(np.array([b.cy for b in rectangles]))
        bottom = np.max(np.array([b.cy for b in rectangles]))
        return left, right, top, bottom

    def get_index(lower, upper, length):
        """Find indeces that fit in the given boundaries.

        Parameters
        ----------
        lower : float
            lower boundary
        upper : float
            upper boundary
        length : float
            num entries of the output array along one axis: either h or w for
            an output array with (h, w)

        Returns
        -------
        float, float
            integer value of all indeces that fit in the give boundary
        """
        # works with top-left coordinates, only scaling
        tmp = lower*length
        index_lower = np.ceil(tmp)
        index_lower = max(index_lower, 0)

        tmp = upper*length
        index_upper = np.floor(tmp)
        index_upper = min(index_upper, length-1)

        return index_lower, index_upper

    def check_if_valid_index(index_lower, index_upper):
        """Check if indeces can be used

        Parameters
        ----------
        index_lower : int or float
        index_upper : int or float

        Returns
        -------
        boolean
            Return True if indeces are feasible
        """
        if index_lower > index_upper or index_upper < index_lower:
            return False
        else:
            return True

    def get_C_rectangle(index_x, index_y, def_box, shape):
        """Return a center rectangle

        Parameters
        ----------
        index_x : int
            index in the output array out(index_y, index_x)
        index_y : int
            index in the output array out(index_y, index_x)
        def_box : Box_spec
            A box specification containing the boxes width and height.
        shape : tuple of int
            shape of the output array

        Returns
        -------
        CenterNormalizedRectangle
            Rectangle in scaled coordinates. out[0, 0] is coordinate (0, 0) and
            out[h-1, w-1] is coordinate(1, 1). Note that the point is shifted
            to the center of the pixel. Therefore e.g. cx = (ix + .5)/w
        """
        default_box = CenterNormalizedRectangle(cx=index_to_norm(index_x, shape[1]),
                                                cy=index_to_norm(
                                                    index_y, shape[0]),
                                                h=def_box.h,
                                                w=def_box.w)
        return default_box

    def get_valid_indeces():
        """Return an array of valid indeces for the current x and y indeces.
        All of those indece pair define a rectangle of the current
        box_definition, that has a JI >= .5 with the current ground truth
        rectangle.

        Returns
        -------
        array, array of shape (n,)
        """
        valid_x = np.arange(x_index_left, x_index_right+1)
        valid_y = np.arange(y_index_top, y_index_bottom+1)
        return valid_x, valid_y

    def set_confidence_map_to_JI():
        """In the unlikely case that a box of the current indeces has a JI >= .5
        with several ground truth boxes, keep the one with the highest JI.

        Returns
        -------
        np.array
            current output conf array
        """
        box_already_set = output_conf[int(index_y), int(index_x), index_z] > 0
        if box_already_set:

            current_JI = output_conf[int(index_y), int(index_x), index_z]

            do_keep_current_box = current_JI > JI
            if do_keep_current_box:
                return do_keep_current_box

        # set the new box value which is
        # temporarily set to jaccard index instead of 1.0
        output_conf[int(index_y), int(index_x), defbox_spec.index] = JI
        return False

    def set_next_best_default_box():
        """For each ground truth box find the anchor box with the highes JI.
        """
        # we need to find the next best match
        x_g, y_g = gt_rectangle.cx, gt_rectangle.cy  # centered, normalized
        max_JI = -np.inf
        for (output_conf, output_loc, prior) in outputs_and_priors:
            shape = prior.shape
            default_boxes = prior.box_specs

            # index that is next to x_g and y_g
            index_x = np.round(x_g*shape[1] - .5)
            index_y = np.round(y_g*shape[0] - .5)
            for def_box in default_boxes:
                index_z = def_box.index

                box_is_already_set = np.any(
                    output_conf[int(index_y), int(index_x), index_z] > 0.)
                if box_is_already_set:
                    # look for the next best open field
                    for dx in [-1, 0, +1]:
                        for dy in [-1, 0, +1]:
                            try:
                                box_is_already_set = np.any(
                                    output_conf[int(index_y + dy),
                                                int(index_x + dx),
                                                index_z] > 0.)
                            except:
                                continue
                            if not box_is_already_set:
                                index_x += dx
                                index_y += dy
                                break
                        if not box_is_already_set:
                            break

                if box_is_already_set:
                    continue

                # centered, normalized
                box = get_C_rectangle(index_x, index_y, def_box, shape)
                # between two c rectangles
                JI = gt_rectangle.estimate_jaccard_index(
                    box, 1.0, 1.0
                )
                if JI > max_JI:
                    max_JI = JI
                    best_output_conf = output_conf
                    best_output_loc = output_loc
                    best_index_x = index_x
                    best_index_y = index_y
                    best_box_index = index_z
                    best_box = box

        # depending on the choice of priors not all rectangles get a fit
        if max_JI <= -np.inf:
            print('unmatched rectangle: {}'.format(gt_rectangle.__dict__))
            return
        # reset used indeces
        index_x = best_index_x
        index_y = best_index_y
        index_z = best_box_index
        # set best match
        best_output_conf[int(index_y), int(index_x), index_z] = 1.0
        # set offset
        set_offsets(index_x, index_y, index_z,
                    best_box, gt_rectangle,
                    best_output_loc)

    def set_offsets(index_x, index_y, index_z,
                    default_box, ground_truth_box,
                    output):
        """Set offset values for a positive rectangle to fit a ground truth
        rectangle

        Parameters
        ----------
        index_x : float
            corresponding pixel in the output feature-map
        index_y : float
            corresponding pixel in the output feature-map
        index_z : int
            coordinate int the output feature-map defining which box is meant
        default_box : CenterNormalizedRectangle
            Hit box that fits the ground truth
        ground_truth_box : CenterNormalizedRectangle
        output : np.array of shape (h,w,4*num_boxes)
            array storing all offset values. The specific order is defined
            in the rectangles class.

        """
        l_x = oc.loc_offset_to_scaled_value(
            ground_truth_box.cx,
            default_box.cx,
            default_box.w
        )

        l_y = oc.loc_offset_to_scaled_value(
            ground_truth_box.cy,
            default_box.cy,
            default_box.h
        )

        l_w = oc.size_factor_to_scaled_value(
            ground_truth_box.w,
            default_box.w
        )

        l_h = oc.size_factor_to_scaled_value(
            ground_truth_box.h,
            default_box.h
        )

        offsets = np.zeros(4)
        offsets[CenterNormalizedRectangle.x_pos] = l_x
        offsets[CenterNormalizedRectangle.y_pos] = l_y
        offsets[CenterNormalizedRectangle.h_pos] = l_h
        offsets[CenterNormalizedRectangle.w_pos] = l_w

        output[int(index_y), int(index_x), index_z*4:(index_z+1)*4] = offsets

    def set_real_confidence_score():
        """During the matching phase JI values are stored in the conf outputs
        to determine to which gt_rectangle the anchor box has a best match.
        Now, those values are set to 1.0 to be used as a label.

        """
        # set jaccard index to confidence score -> 1.0
        for i, (out_conf, _, _) in enumerate(outputs_and_priors):
            outputs_and_priors[i][0] = np.ceil(out_conf)

    def add_negative_class():
        """To be used by the keras cross-entropy function. The negative class
        needs to be added to each anchor-box's confidence label.

        """
        for j, (out_conf, _, _) in enumerate(outputs_and_priors):
            h, w, d = out_conf.shape
            neg_pos_conf_output = np.zeros((h, w, 2*d))
            for i in range(d):
                pos = out_conf[..., i]
                neg = 1.0 - pos
                neg_pos_conf_output[..., 2*i] = neg
                neg_pos_conf_output[..., 2*i+1] = pos
            outputs_and_priors[j][0] = neg_pos_conf_output

    def get_flattened_vector():
        """Return a vector of shape (n, ). For each shape,
        the order is conf_0, conf_1, ..., conf_n, loc_0, ..., loc_n
        Shapes are then concatenated. Hence,
        s0_c0, ... , s0_cn, s0_l0, ..., s0_ln, s1_c0, ..., s1_cm, s1_l0, ...,
        s1_lm, s2_c0, ...


        Returns
        -------
        np.array
            ssd label vector
        """
        output = []
        for (conf, loc, _) in outputs_and_priors:
            out = np.concatenate([conf, loc], axis=-1)
            # print(out.shape)
            output.append(out.flatten())
        # if you reshape here it works
        try:
            output_flat = np.concatenate(output)
        except Exception as identifier:
            print('!'*10)
            print(identifier)
            print(outputs_and_priors)
            print(output)
            print('!'*10)
            raise ValueError('Concat problem.')
        return output_flat

    # -----------------start of logic------------------------------
    gt_rectangles = get_normalized_rectangles(label)  # centered, normalized

    outputs_and_priors = setup_outputs_and_priors()
    # ensure that each rectangle has at least one match
    # those are set to 1.0 and will not be overwritten

    for gt_rectangle in gt_rectangles:
        set_next_best_default_box()  # input: centered, normalized

    # also set default boxes that land a JI > .5
    # in this area of the code, gt_rectangles need to be top-left rectangles

    for gt_rectangle in gt_rectangles:
        # top-left, normalized
        gt_rectangle = gt_rectangle.get_normalized_top_left()
        for (output_conf, output_loc, prior) in outputs_and_priors:
            shape = prior.shape
            defbox_specs = prior.box_specs

            for defbox_spec in defbox_specs:
                rectangles = find_pointsII(
                    gt_rectangle, defbox_spec.w, defbox_spec.h)
                if not rectangles:
                    continue

                left, right, top, bottom = get_valid_area()

                x_index_left, x_index_right = get_index(
                    left, right, shape[1])
                if not check_if_valid_index(x_index_left, x_index_right):
                    continue

                y_index_top, y_index_bottom = get_index(
                    top, bottom, shape[0])
                if not check_if_valid_index(y_index_top, y_index_bottom):
                    continue

                valid_x, valid_y = get_valid_indeces()
                # ! indeces found here are for N_rectangles (x, y are topleft)
                for index_x in list(valid_x):
                    for index_y in list(valid_y):
                        index_z = defbox_spec.index

                        # centered, normalized
                        default_box = get_C_rectangle(
                            index_x, index_y, defbox_spec, shape)

                        # JI needed when there are multiple boxes fitting the
                        # indeces. Only the box with the highest JI is used.

                        JI = gt_rectangle.estimate_jaccard_index(
                            # top-left, normalized
                            default_box.get_normalized_top_left()
                        )

                        if JI < .5:
                            continue
                        box_is_already_set = set_confidence_map_to_JI()

                        if not box_is_already_set:
                            # centered, normalized
                            ground_truth_box = gt_rectangle.to_c_rectangle(
                                1.0, 1.0)
                            set_offsets(index_x, index_y, index_z,
                                        default_box, ground_truth_box,
                                        output_loc)

    set_real_confidence_score()
    add_negative_class()

    return get_flattened_vector()


def find_pointsII(gt_rectangle, w_p, h_p):
    """Find the rectangles at the top-left most-, bottom-left most, top-right 
    most- and bottom-right most corners that have a JI equal to 1/2 to the 
    ground truth rectangle.

    ! The rectangle needs to be a top-left Rectangle for this function!

    This is done with lagrange-multipliers. For the top-left most rectangle,
    for the rest, symmetries are used.
    The Lagrange function is:
    f(x,y) := -x -y
    s.t.
    g(x,y) := I - A/3 = 0 (I/(A - I) = 1/2)
    with Intersection I = (x + w_p - x_g)(y + h_p - y_g)
    Sum of Area A = h_p*w_p + h_g*w_g
    _p is prediction, _g ground truth. 
    A quadratic equation needs to be solved.

    Parameters
    ----------
    gt_rectangle : Rectangle (top-left)
        box for which the coordinates are computed
    w_p : float
        width of the anchor box that is fitted
    h_p : float
        height of the anchor box that is fitted

    Returns
    -------
    list of rectangles
        If a solution exists return the four rectangles that fit the gt-box
        else, return an empty list.
    """
    x_g, y_g = gt_rectangle.x, gt_rectangle.y
    w_g, h_g = gt_rectangle.w, gt_rectangle.h

    # check if solution possible. If one rectangle encloses the other,
    # the JI must not be < 1/2.
    A_g = w_g*h_g
    A_p = w_p*h_p

    has_no_solution = A_g/A_p < .5 or A_g/A_p > 2.0
    if has_no_solution:
        return []

    A = A_g + A_p

    # setup quadratic coefficients x**2 + bx + c = 0
    b = 2.0*(w_p - x_g)
    c = -2.0*x_g*w_p + w_p**2 + x_g**2 - A/3.0

    # solve quadratic equation
    x = .5*(-b + np.sqrt(b**2 - 4*c))

    y = x + w_p - x_g - h_p + y_g
    # generate alternative solutions for right coordinate and bottom
    x_r = 2.0*x_g - (w_p - w_g) - x
    y_b = 2.0*y_g - (h_p - h_g) - y

    solutions = []
    for u in [x, x_r]:
        for v in [y, y_b]:

            rect = TopLeftRectangle(
                x=u,
                y=v,
                w=w_p,
                h=h_p
            )
            solutions.append(
                #centered, normalized
                rect.to_c_rectangle(1.0, 1.0)
            )
    return solutions


def get_normalized_rectangles(cc_image, h_norm=None, w_norm=None):
    """Compute a list of rectangles enclosing the instances in the given
    connected-components image

    Take special care of which norm-factors are used to compute the scaled
    coordinates.

    Parameters
    ----------
    cc_image : np.array
        each integer describes to which instance a pixel belongs
    h_norm : float, optional
        factor to scale the rectangle. (The default is None, in this case,
        the image height is used. Hence all values range between 0 and 1)
    w_norm : [type], optional
        factor to scale the rectangle. (The default is None, in this case,
        the image height is used. Hence all values range between 0 and 1)



    Returns
    -------
    list of cRectangles
        Each rectangle exactly fits its instance.
    """
    if h_norm is None and w_norm is None:
        h, w = cc_image.shape[0], cc_image.shape[1]
    else:
        h, w = h_norm, w_norm

    stats = compute_connected_component_stats(cc_image)
    rectangles = []
    for stat in stats:
        stat[cv2.CC_STAT_LEFT] = (
            float(stat[cv2.CC_STAT_LEFT])
        )/float(w)

        stat[cv2.CC_STAT_TOP] = (
            float(stat[cv2.CC_STAT_TOP])
        )/float(h)

        stat[cv2.CC_STAT_HEIGHT] = float(
            stat[cv2.CC_STAT_HEIGHT])/float(h)

        stat[cv2.CC_STAT_WIDTH] = float(
            stat[cv2.CC_STAT_WIDTH])/float(w)

        rectangle = TopLeftRectangle(
            x=stat[cv2.CC_STAT_LEFT],
            y=stat[cv2.CC_STAT_TOP],
            h=stat[cv2.CC_STAT_HEIGHT],
            w=stat[cv2.CC_STAT_WIDTH],
            # h_norm=float(h),
            # w_norm=float(w)
        )
        # rectangle is already scaled
        # centered and normalized rectangle
        rectangles.append(rectangle.to_c_rectangle(1.0, 1.0))

    return rectangles


def setup_priors(shapes, heights, a_ratios):
    """Make a list of priors from three lists: 
    shapes, heights and a_ratios.
    With shapes[i], heights[i] and a_ratios[i] one anchor box is defined.

    Returns
    -------
    list of priors
        A prior is a tuple with its output shape (h,w) and a list of box_specs.
    """
    # put same shapes together
    d = dict()
    for s, h, ar in zip(shapes, heights, a_ratios):
        box_specs = []
        w = h*ar
        box_specs.append(
            BoxSpec(
                index=-1,
                h=h,
                w=w
            )
        )
        if ar != 1.0:
            # NOTE: switch height with width
            # old version: w = h/ar
            box_specs.append(
                BoxSpec(
                    index=-1,
                    h=w,
                    w=h
                )
            )

        if s not in d:
            d[s] = box_specs
        else:
            d[s].extend(box_specs)

    # shapes need to be in specific order!
    priors = []
    keys = d.keys()
    # python2: keys.sort()
    sorted(keys)
    for key in keys:
        val = d[key]
        shape = (key, key)
        # print(shape)
        new_val = []
        for i, v in enumerate(val):
            new_val.append(
                BoxSpec(index=i,
                        h=v.h,
                        w=v.w)
            )
            #print(v.h, v.w)
        priors.append(Prior(shape=shape, box_specs=new_val))
    return priors


def index_to_norm(i, s):
    """Compute the normalized and centered coordinate of an index.

    Parameters
    ----------
    i : float or int
        integer index
    s : int
        shape of one axis. Eg x = index_to_norm(x_index, w)

    Returns
    -------
    float
        normalized and centerd coordinate of the given index when the feature
        map has the given axis-size.
    """
    return (float(i) + .5)/float(s)


def norm_to_index(z, s):
    """Compute the index from a normalized and centered coordinate.

    Parameters
    ----------
    z : float
        normalized and centered coordinate
    s : int
        shape of one axis. Eg x_index = norm_to_index(x, w) 

    Returns
    -------
    float
        float value of the index.
    """
    return z*float(s) - .5
