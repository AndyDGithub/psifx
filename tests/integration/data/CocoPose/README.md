# CocoPose — COCO val2017 test subset

8 images from COCO val2017 with their `person_keypoints_val2017` annotations
(one annotated person per image, picked as the 8 images where that person's
bounding box covers the largest share of the frame, out of the 50-image
benchmark subset produced by `research/benchmark/prepare_coco_subset.py` on the dev branch).

Used by `tests/integration/video/test_pose_coco_accuracy.py`.

COCO annotations are licensed CC BY 4.0; images keep their original Flickr
licenses (see the `license` field of each image and https://cocodataset.org/#termsofuse).
