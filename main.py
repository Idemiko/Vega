import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from pixellib.instance import instance_segmentation


def object_detection_on_an_image():
    segment_image = instance_segmentation()
    segment_image.load_model(".venv/mask_rcnn_balloon.h5")

    target_class = segment_image.select_target_classes(person=True)

    result = segment_image.segmentImage(
        image_path="1street.jpg",
        segment_target_classes=target_class,

    )

    # print(result[0]["scores"])
    objects_count = len(result[0]["scores"])
    print(f"Найдено объектов: {objects_count}")


def main():
    object_detection_on_an_image()


if __name__ == '__main__':
    main()