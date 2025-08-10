from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rfstudio.engine.task import Task
from rfstudio.graphics import TriangleMesh
from rfstudio.visualization import Visualizer
from rfstudio.utils.colormap import RainbowColorMap


@dataclass
class Script(Task):

    base_dir: Path = Path('exp_results/clustering/artgs')
    name: str = 'blade'
    viser: Visualizer = Visualizer()

    def run(self) -> None:
        parts = list((self.base_dir / 'ply').glob(f'{self.name}_0_*.ply'))

        with self.viser.customize() as handle:
            colors = RainbowColorMap()(torch.linspace(0, 1, len(parts)).unsqueeze(-1))
            for i, part in enumerate(parts):
                mesh = TriangleMesh.from_file(part)
                handle[f'part_{i:02d}'].show(mesh).configurate(vertex_colors=colors[i].expand_as(mesh.vertices))

if __name__ == '__main__':
    Script(cuda=0).run()
