from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.graphics import TriangleMesh
from rfstudio.ui import console


@dataclass
class Script(Task):

    input: Path = ...
    output: Path = ...
    max_num_faces: int = 8192

    def run(self) -> None:
        assert self.input.exists()
        if self.input.is_dir():
            inputs = list(self.input.glob("*.ply")) + list(self.input.glob("*.obj"))
        else:
            assert self.input.suffix in ['.obj', '.ply']
            inputs = [self.input]

        for input_mesh in inputs:
            output = self.output
            if output.exists():
                if output.is_file():
                    assert output.suffix in ['.obj', '.ply']
                    assert len(inputs) == 1
                else:
                    output = output / input_mesh.name
            else:
                if output.suffix in ['.obj', '.ply']:
                    assert len(inputs) == 1
                    output.parent.mkdir(exist_ok=True, parents=True)
                else:
                    output.mkdir(exist_ok=True, parents=True)
                    output = output / input_mesh.name
            mesh = TriangleMesh.from_file(input_mesh)
            simpl_mesh = mesh.simplify(self.max_num_faces / mesh.num_faces)
            simpl_mesh.export(output, only_geometry=True)
            console.print(f'{input_mesh} -> {output}')
            console.print(f'  #{mesh.num_faces} -> #{simpl_mesh.num_faces} ({simpl_mesh.num_faces / mesh.num_faces:.1%})')


if __name__ == '__main__':
    Script(cuda=0).run()
