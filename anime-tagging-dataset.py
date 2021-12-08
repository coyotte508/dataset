from pathlib import Path
from typing import List

import datasets
from datasets.tasks import ImageClassification


logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = "Anime tagging."

tags = ['long_hair', 'original', 'headphones', 'open_mouth', 'hoshino_katsura', 'brown_hair', 'shoes', 'elbow_gloves', 'flower', 'glasses', 'bow', 'sitting', 'scan', 'yellow_eyes', 'hatsune_miku', 'uniform', 'suzumiya_haruhi_no_yuuutsu', 'purple_hair', 'flat_chest', 'thigh-highs', 'ponytail', 'vector', 'ribbon', 'blonde_hair', 'cat_ears', 'red_eyes', '1girl', 'hair_ornament', 'hat', 'sky', 'seifuku', 'water', 'smile', 'brown_eyes', 'bad_id', 'twintails', 'black_hair', 'pink_hair', 'vocaloid', 'ahoge', 'red_hair', 'tagme', 'monochrome', 'cleavage', 'nail_polish', 'hair_ribbon', 'weapon', 'wings', 'legs', 'necktie', 'touhou', 'zettai_ryouiki', 'dress', 'purple_eyes', 'solo', 'closed_eyes', 'catgirl', 'male', 'blue_hair', 'midriff', 'navel', 'white_hair', 'skirt', 'aqua_hair', 'scarf', 'jewelry', 'trap', 'kagamine_len', 'animal_ears', 'very_long_hair', 'wink', 'pantyhose', 'socks', 'green_eyes', 'nagato_yuki', 'sword', 'thigh_highs', 'boots', 'swimsuit', 'd.gray-man', 'wallpaper', 'blue_eyes', 'blush', 'detached_sleeves', 'bikini', 'short_hair', 'japanese_clothes', 'kimono', 'thighhighs', 'green_hair', 'hair_bow', 'multiple_girls', 'tail', 'highres', 'breasts', 'white', 'school_uniform', 'gloves', 'megurine_luka', 'itou_noiji']


class AnimeTagging(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image_file_path": datasets.Value("string"),
                    "labels": datasets.features.ClassLabel(names=tags),
                }
            ),
            supervised_keys=("image_file_path", "labels"),
            task_templates=[
                ImageClassification(
                    image_file_path_column="image_file_path", label_column="labels", labels=tags
                )
            ],
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # images_path = Path(dl_manager.download_and_extract(_URL)) / "PetImages"
        images_path = Path(".")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"images_path": images_path}),
        ]

    def _generate_examples(self, images_path):
        print("generating examples from = %s", images_path)
        logger.info("generating examples from = %s", images_path)
        for i, filepath in enumerate(images_path.glob("**/*.jpg")):
            with filepath.open("rb") as f:
                if b"JFIF" in f.peek(10):
                    yield str(i), {
                        "image_file_path": str(filepath),
                        "labels": filepath.parent.name.lower(),
                    }
                    continue