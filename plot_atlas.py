import collections.abc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nilearn import datasets, image
from nilearn._utils import check_niimg_3d
from nilearn._utils.niimg import safe_get_data
from typing import Any
from nilearn.plotting.matrix_plotting import plot_matrix
import matplotlib.pyplot as plt

DEBUG = False

NORMALIZE = True

THRESHOLD = 0.01

# HOA
dataset_ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
# "cort-maxprob-thr0-1mm"
# "cort-maxprob-thr25-1mm"
# "cort-maxprob-thr50-1mm"
# "sub-maxprob-thr0-1mm"
# "sub-maxprob-thr25-1mm"
# "sub-maxprob-thr50-1mm"

#  Juelich
dataset_ju = datasets.fetch_atlas_juelich("maxprob-thr25-1mm")
# "maxprob-thr0-1mm"
# "maxprob-thr25-1mm"
# "maxprob-thr50-1mm"

dataset_des = datasets.fetch_atlas_destrieux_2009()

dataset_tal = datasets.fetch_atlas_talairach('lobe') # hemisphere lobe gyrus tissue ba

dataset_pauli = datasets.fetch_atlas_pauli_2017(version='det')

source_atlas = {"name": "HOA", "image": dataset_ho.maps, "labels": dataset_ho.labels}
target_atlas = {"name": "juelich", "image": dataset_ju.maps, "labels": dataset_ju.labels}

def plot_sankey_diagram(source, target, value, all_labels, color, title):
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=100,
                    thickness=100,
                    line=dict(color="black", width=0.5),
                    label=all_labels,
                    color=color,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )

    fig.update_layout(title_text=title, font_size=20)

    fig.show()


def main(source_atlas: dict[str, Any], target_atlas, threshold: int = 0):

    LABELS_TO_SKIP = ["Background"]

    labels_img = check_niimg_3d(source_atlas["image"])
    labels_data = safe_get_data(source_atlas["image"], ensure_finite=True)

    nb_labels_source = len(source_atlas["labels"])
    nb_labels_target = len(target_atlas["labels"])

    check_unique_labels = np.unique(labels_data)
    if np.any(check_unique_labels < 0):
        raise ValueError(
            "The 'labels_img' you provided has unknown/negative "
            f"integers as labels {check_unique_labels} assigned to regions. "
            "All regions in an image should have positive "
            "integers assigned as labels."
        )

    unique_labels = set(check_unique_labels)

    if source_atlas["labels"] is not None:
        if not isinstance(source_atlas["labels"], collections.abc.Iterable) or isinstance(
            source_atlas["labels"], str
        ):
            source_atlas["labels"] = [source_atlas["labels"]]
        if len(unique_labels) != nb_labels_source:
            raise ValueError(
                f"The number of labels: {nb_labels_source} provided as input "
                f"in labels={source_atlas['labels']} does not match with the number "
                f"of unique labels in labels_img: {len(unique_labels)}. "
                "Please provide appropriate match with unique "
                "number of labels in labels_img."
            )

    #  resample target atlas to source atlas
    target_data = image.resample_img(
        target_atlas["image"],
        target_affine=labels_img.affine,
        target_shape=labels_img.shape[:3],
        interpolation="nearest",
        copy=True,
    )

    title = f"{source_atlas['name']} VS {target_atlas['name']}"


    matrix = np.zeros((nb_labels_source, nb_labels_target))

    source = []
    target = []
    value = []

    for label_id, name in zip(unique_labels, source_atlas["labels"]):

        if name in LABELS_TO_SKIP:
            continue

        this_label_mask = labels_data == label_id

        voxel_values = target_data.get_fdata()[this_label_mask]

        targets = pd.Series(voxel_values).value_counts()
        for l, v in targets.items():
            if NORMALIZE:
                v = v / len(voxel_values)
            if v > threshold:
                if target_atlas["labels"][int(l)] in LABELS_TO_SKIP:
                    continue
                source.append(label_id)
                target.append(int(l) + nb_labels_source)
                value.append(v)

                matrix[label_id, int(l)] = v

    all_labels = source_atlas["labels"] + target_atlas["labels"]

    color = ["blue" for x in source_atlas["labels"]] + ["red" for x in target_atlas["labels"]]

    fig, (ax1) = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
    plot_matrix(
    matrix,
    title=title,
    labels=None,
    axes=ax1,
    colorbar=True,
    cmap=plt.get_cmap('gray'),
    tri="full",
    auto_fit=True,
    grid=False,
    reorder=False
)
    fig.savefig('tmp.png')

    plot_sankey_diagram(source, target, value, all_labels, color, title)


if __name__ == "__main__":
    main(source_atlas, target_atlas, threshold=THRESHOLD)
