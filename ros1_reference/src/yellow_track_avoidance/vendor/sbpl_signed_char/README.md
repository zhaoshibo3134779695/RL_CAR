# Package-local SBPL runtime

License: BSD 3-Clause; see `LICENSE`. The included binary must match the host
architecture. Run `../../tools/build_sbpl_runtime.sh` after moving the package
to a different architecture and before starting `move_base`.

`lib/libsbpl.so` is SBPL 1.3.1 (upstream commit
`4d654845aa3b92aa1b4282a342e9011bac95aeb9`) compiled with
`-fsigned-char`.

SBPL 1.3.1 stores signed XYTheta lattice `dX` and `dY` action offsets in plain
`char` fields. The explicit compiler option gives every target the same
semantics and avoids AArch64's unsigned-plain-char failure mode.

Only the `move_base` nodes in this package prepend this directory to
`LD_LIBRARY_PATH`; the system copy under `/opt/ros/noetic` is unchanged.

Rebuild with:

```bash
./tools/build_sbpl_runtime.sh
```
