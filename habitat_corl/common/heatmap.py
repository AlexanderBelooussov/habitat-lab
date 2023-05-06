import argparse

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from habitat_baselines.config.default import get_config
from habitat_corl.shortest_path_dataset import load_full_dataset, \
    get_stored_groups
from habitat_corl.train import scene_dict, dataset_dict


def load_datasets(scene):
    config_path = "habitat_corl/configs/sacn_pointnav.yaml"
    config = get_config(config_path, [])
    config.defrost()
    config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_dict[scene]]
    config.TASK_CONFIG.DATASET.SP_DATASET_PATH = dataset_dict[scene]
    datasets = []
    for (web_dataset, no_sp) in [(False, False), (True, True)]:
        if web_dataset:
            config.TASK_CONFIG.DATASET.WEB_DATASET_PATH = f"data/web_datasets/web_dataset_{scene}.hdf5"
        if no_sp:
            config.TASK_CONFIG.DATASET.SP_DATASET_PATH = None
            groups = None
        else:
            groups = get_stored_groups(
                config.TASK_CONFIG.DATASET.SP_DATASET_PATH)[
                     :1000]

        dataset = load_full_dataset(config.TASK_CONFIG,
                                    datasets=["states/position"])
        datasets.append(dataset)
    return datasets[0], datasets[1]


def plot_heatmap(sp_ds, web_ds, scene, web_dataset, no_sp):
    if web_dataset and not no_sp:
        positions = web_ds.states["position"]
        positions = np.append(positions, sp_ds.states["position"], axis=0)
    elif web_dataset:
        positions = web_ds.states["position"]
    elif not no_sp:
        positions = sp_ds.states["position"]

    if scene in ["medium", "large"]:
        x = positions[:, 0]
        y = positions[:, 2]
    else:
        x = positions[:, 2]
        y = positions[:, 0]

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=128)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # replace 0 with None
    heatmap[heatmap == 0] = None

    plt.clf()
    im = plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm())
    ax = plt.gca()
    if scene in ["medium", "large"]:
        ax.invert_yaxis()

    # label axes
    if scene in ["medium", "large"]:
        plt.xlabel('x')
        plt.ylabel('z')
    else:
        plt.xlabel('z')
        plt.ylabel('x')
    # add title
    if not no_sp and web_dataset:
        datasets = "shortest path + habitat-web"
    elif not no_sp:
        datasets = "shortest path"
    elif web_dataset:
        datasets = "habitat-web"
    plt.title(
        f'Heatmap of positions in {datasets} dataset\n{scene.capitalize()} scene')

    # add legend
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(
        f"/home/alexb/Downloads/heatmap_{scene}_{'web' if web_dataset else 'no_web'}_{'no_sp' if no_sp else 'sp'}.png"
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--web_dataset",
        action="store_true",
        help="Use web dataset",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="medium",
        choices=["medium", "large", "small", "xl", "debug"],
        help="Scene to use",
    )
    parser.add_argument(
        "--no_sp",
        action="store_true",
        help="Don't use shortest path dataset",
    )

    args = parser.parse_args()
    web_dataset = args.web_dataset
    scene = args.scene
    no_sp = args.no_sp

    plot_heatmap(scene, web_dataset, no_sp)


if __name__ == "__main__":
    for scene in ["small", "medium", "large", "xl"]:
        sp, web = load_datasets(scene)
        for web_dataset in [True, False]:
            plot_heatmap(sp, web, scene, web_dataset, False)
        plot_heatmap(sp, web, scene, True, True)
