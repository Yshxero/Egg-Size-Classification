import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.measure import label, regionprops


def segment_egg(img):

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(th == 255) < np.sum(th == 0):
        mask = th
    else:
        mask = cv2.bitwise_not(th)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    white_pixels = np.sum(mask == 255)
    if white_pixels < 500:
        return None

    num_labels, labels_im = cv2.connectedComponents(mask)
    if num_labels <= 1:
        return mask

    areas = [(labels_im == i).sum() for i in range(1, num_labels)]
    largest_i = np.argmax(areas) + 1
    clean_mask = (labels_im == largest_i).astype('uint8') * 255

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    if len(cnts) == 0:
        return None

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [cnts[0]], -1, 255, -1)

    return clean_mask

def extract_geometric_features(mask):
    mask_bool = mask.astype(bool)
    lbl = label(mask_bool)
    props = regionprops(lbl)
    if len(props) == 0:
        return {k: np.nan for k in ['area', 'perimeter', 'maj_axis', 'min_axis',
                                    'eccentricity', 'equiv_diameter', 'circularity',
                                    'solidity', 'extent']}
    p = props[0]
    area = p.area
    perimeter = p.perimeter if p.perimeter > 0 else 1.0
    maj = max(p.major_axis_length, p.minor_axis_length)
    min_ax = min(p.major_axis_length, p.minor_axis_length)
    eccentricity = p.eccentricity
    equiv_diam = p.equivalent_diameter
    circularity = 4 * np.pi * area / (perimeter ** 2)
    convex_area = p.convex_area if hasattr(p, 'convex_area') else area
    solidity = area / convex_area if convex_area > 0 else 0
    extent = p.extent
    return {
        'area': area,
        'perimeter': perimeter,
        'maj_axis': maj,
        'min_axis': min_ax,
        'eccentricity': eccentricity,
        'equiv_diameter': equiv_diam,
        'circularity': circularity,
        'solidity': solidity,
        'extent': extent
    }

def extract_lbp_features(gray, mask, P=8, R=1, n_bins=10):
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    masked_lbp = lbp[mask.astype(bool)]
    if masked_lbp.size == 0:
        return np.zeros(n_bins)
    max_val = int(masked_lbp.max() if masked_lbp.size else P + 2)
    hist, _ = np.histogram(masked_lbp, bins=n_bins, range=(0, max_val + 1))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-9)
    return hist

def extract_features_from_image(img):
    mask = segment_egg(img)
    geo = extract_geometric_features(mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp_hist = extract_lbp_features(gray, mask, P=8, R=1, n_bins=10)
    features = geo.copy()
    for i, v in enumerate(lbp_hist):
        features[f'lbp_{i}'] = v
    return features