"""Locate a visually similar planar packaging reference as a review cue."""
import hashlib


def locate(image_path, reference_path):
    import cv2
    import numpy as np
    from PIL import Image

    digest = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    result = {"status": "无法判断", "reference_sha256": digest, "regions": [],
              "reason": "未能可靠定位相似图案；这不代表图片中没有该产品。"}
    with Image.open(image_path) as image, Image.open(reference_path) as reference:
        target_size = image.size
        image.thumbnail((1600, 1600))
        reference.thumbnail((1600, 1600))
        target = np.array(image.convert("L"))
        template = np.array(reference.convert("L"))
    detector = cv2.ORB_create(nfeatures=2000)
    target_points, target_desc = detector.detectAndCompute(target, None)
    ref_points, ref_desc = detector.detectAndCompute(template, None)
    if target_desc is None or ref_desc is None or len(target_desc) < 2:
        return result
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ref_desc, target_desc, k=2)
    matches = [first for pair in pairs if len(pair) == 2 for first, second in [pair]
               if first.distance < 0.7 * second.distance]
    if len(matches) < 12:
        return result
    src = np.float32([ref_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([target_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    transform, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if transform is None or inliers is None or int(inliers.sum()) < 10 or inliers.mean() < 0.6:
        return result
    height, width = template.shape
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    polygon = cv2.perspectiveTransform(corners, transform).reshape(-1, 2)
    h, w = target.shape
    if (not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon.astype(np.float32))
            or cv2.contourArea(polygon.astype(np.float32)) < 0.005 * w * h
            or (polygon[:, 0] < 0).any() or (polygon[:, 0] > w).any()
            or (polygon[:, 1] < 0).any() or (polygon[:, 1] > h).any()):
        return result
    x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
    x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
    bbox = [round(x1 * target_size[0] / w), round(y1 * target_size[1] / h),
            round(x2 * target_size[0] / w), round(y2 * target_size[1] / h)]
    result.update(status="发现相似图案", regions=[{"bbox": bbox}],
                  reason="这是图案定位线索，不能确认产品、色号、真实性或功效；请人工对照参考图。")
    return result
