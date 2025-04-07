# fiftyone_utils.py

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.core.labels as fol
from config import LOW_CONF_THRESHOLD
from tqdm import tqdm

import random

BDD100K_CLASSES = [
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign", "train", "truck"
]

def add_dummy_predictions(dataset, num_preds=3):
    for sample in tqdm(dataset):
        detections = []
        for _ in range(num_preds):
            label = random.choice(BDD100K_CLASSES)
            bbox = [
                round(random.uniform(0.2, 0.6), 2),  # x
                round(random.uniform(0.2, 0.6), 2),  # y
                round(random.uniform(0.1, 0.3), 2),  # w
                round(random.uniform(0.1, 0.3), 2)   # h
            ]
            confidence = round(random.uniform(0.2, 0.9), 2)

            detections.append(fol.Detection(
                label=label,
                bounding_box=bbox,
                confidence=confidence,
            ))

        sample["predictions"] = fol.Detections(detections=detections)
        sample.save()


def built_in_bbk100k(source_local_dir):
    # The path to the source files that you manually downloaded
    dataset = foz.load_zoo_dataset(
        "bdd100k",
        source_dir=source_local_dir,
    )
    session = fo.launch_app(dataset)


def setup_dataset(fiftyone_dataset_zoo_name_or_url, source_local_dir, save_local_dataset_name):
    # 1. 데이터셋 다운로드 및 등록
    print(f"📦 {fiftyone_dataset_zoo_name_or_url} 데이터셋을 로딩 중...")
    dataset = foz.load_zoo_dataset(
        fiftyone_dataset_zoo_name_or_url,
        source_dir=source_local_dir,
        dataset_name=save_local_dataset_name,
        persistent=True,
        overwrite=True,
    )

    print(f"등록 완료: {dataset.name}")
    print(f"총 샘플 수: {len(dataset)}")


def preview_samples(dataset_name, max_samples=3):
    dataset = fo.load_dataset(dataset_name)

    print(f"\n'{dataset_name}' 샘플 미리보기:")
    for sample in dataset[:max_samples]:
        print("파일:", sample.filepath)
        print("사용 가능한 필드:", sample.field_names)

        # 라벨 필드 자동 탐색 및 개수 출력
        label_fields = [f for f in sample.field_names if isinstance(sample[f], fol.Detections)]
        if label_fields:
            for field in label_fields:
                print(f"라벨 필드: '{field}', 객체 수: {len(sample[field].detections)}")
        else:
            print("라벨 없음")

        print("-" * 40)


def launch_dataset_gui(dataset_name):
    dataset = fo.load_dataset(dataset_name)
    print(f"\n{dataset_name} GUI 실행 중...")
    session = fo.launch_app(dataset)
    session.wait()




def load_filtered_samples(dataset_name):
    dataset = fo.load_dataset(dataset_name)

    # 모든 샘플에 'predictions' 필드가 있는지 확인
    predictions_exist = all("predictions" in sample.field_names for sample in dataset)

    if not predictions_exist:
        print("'predictions' 필드가 없어 dummy 예측을 추가합니다...")
        add_dummy_predictions(dataset)

    # 'confidence'가 없는 경우도 대비
    view = dataset.filter_labels(
        "predictions",
        (fo.ViewField("confidence") < LOW_CONF_THRESHOLD) & (fo.ViewField("confidence") != None)
    )

    return view



def get_local_dataset_lists():
    return fo.list_datasets()


def get_local_dataset_infor(local_dataset_name):
    dataset = fo.load_dataset(local_dataset_name)
    return dataset


def delete_local_dataset(local_dataset_name):
    fo.delete_dataset(local_dataset_name)


def verified_local_dataset():
    for local_dataset_name in get_local_dataset_lists():
        data = get_local_dataset_infor(local_dataset_name=local_dataset_name)
        if len(data) == 0:
            delete_local_dataset(local_dataset_name)


if __name__ == "__main__":
    # verified_local_dataset()

    print(get_local_dataset_lists())
    launch_dataset_gui('bdd100k_custom')
