#!/usr/bin/env python3
"""Read an XYZ file and visualize the molecule in 3D using matplotlib.

Default behavior: read 'water_trimer.xyz' from the current working directory and display
an interactive 3D scatter (O in red, H in light gray). Use --save to save a PNG file
instead of (or in addition to) showing the plot.
"""
import os
import sys
import argparse
import numpy as np


def parse_xyz(path):
    """Parse a simple XYZ file and return (symbols, coords) where coords is an (N,3) ndarray."""
    with open(path, 'r') as f:
        lines = [l.rstrip() for l in f.readlines()]
    if len(lines) < 3:
        raise ValueError('XYZ file appears too short')
    # skip first two lines (atom count and comment)
    atom_lines = [l for l in lines[2:] if l.strip()]
    symbols = []
    coords = []
    for ln in atom_lines:
        parts = ln.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f'Could not parse coordinates from line: "{ln}"')
        symbols.append(sym)
        coords.append((x, y, z))
    if len(symbols) == 0:
        raise ValueError('No atoms parsed from XYZ file')
    return symbols, np.array(coords)


def _set_equal_aspect(ax, X, Y, Z):
    # Prefer set_box_aspect when available (matplotlib >= 3.3)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        # Fallback: set limits to equal ranges
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        xmid = 0.5 * (X.max() + X.min())
        ymid = 0.5 * (Y.max() + Y.min())
        zmid = 0.5 * (Z.max() + Z.min())
        ax.set_xlim(xmid - max_range/2, xmid + max_range/2)
        ax.set_ylim(ymid - max_range/2, ymid + max_range/2)
        ax.set_zlim(zmid - max_range/2, zmid + max_range/2)


def plot_molecule(symbols, coords, save_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Separate atom types
    idx_O = [i for i, s in enumerate(symbols) if s.upper().startswith('O')]
    idx_H = [i for i, s in enumerate(symbols) if s.upper().startswith('H')]

    # Draw bonds first so atom markers appear on top
    # O-H bonds: assign each hydrogen to its nearest oxygen (one-to-one assignment per H)
    if len(idx_O) > 0 and len(idx_H) > 0:
        O_coords = coords[idx_O]
        H_coords = coords[idx_H]
        # For each H, find nearest O
        assigned = {o_idx: [] for o_idx in idx_O}
        for h_local_idx, h_idx in enumerate(idx_H):
            h_coord = coords[h_idx]
            dists = np.linalg.norm(O_coords - h_coord, axis=1)
            oi = int(np.argmin(dists))
            o_idx = idx_O[oi]
            # Assign H to nearest O (no cutoff) so each H connects to exactly one O
            assigned[o_idx].append(h_idx)
        # Now draw O-H bonds for assigned hydrogen indices
        for o_idx, h_list in assigned.items():
            O_coord = coords[o_idx]
            # If more than two Hs were assigned erroneously, keep the two closest
            if len(h_list) > 2:
                # sort by distance and keep two
                h_list = sorted(h_list, key=lambda hi: np.linalg.norm(coords[hi] - O_coord))[:2]
            for h_idx in h_list:
                hx, hy, hz = coords[h_idx]
                ox, oy, oz = O_coord
                ax.plot([ox, hx], [oy, hy], [oz, hz], color='lightgray', linewidth=1.0, zorder=1)

    # NOTE: O-O backbone lines removed per user request (no O-O bonds drawn).

    # Now draw atom markers on top of bonds
    if len(idx_O) > 0:
        ax.scatter(X[idx_O], Y[idx_O], Z[idx_O], c='red', s=220, edgecolors='k', label='O', zorder=10)
    if len(idx_H) > 0:
        ax.scatter(X[idx_H], Y[idx_H], Z[idx_H], c='lightgray', s=60, edgecolors='k', label='H', zorder=10)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Water Trimer 3D Structure')

    _set_equal_aspect(ax, X, Y, Z)

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f'Saved visualization to: {save_path}')

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize water_trimer.xyz in 3D')
    parser.add_argument('--file', '-f', default='water_trimer.xyz',
                        help='Path to XYZ file (default: water_trimer.xyz in cwd)')
    parser.add_argument('--save', '-s', nargs='?', const='water_trimer.png', default=None,
                        help='If provided, save figure to this path (default name when flag used without arg: water_trimer.png)')
    args = parser.parse_args()

    xyz_path = args.file
    if not os.path.isabs(xyz_path):
        # resolve relative to current working directory
        xyz_path = os.path.join(os.getcwd(), xyz_path)

    if not os.path.exists(xyz_path):
        print(f'ERROR: XYZ file not found: {xyz_path}', file=sys.stderr)
        sys.exit(2)

    try:
        symbols, coords = parse_xyz(xyz_path)
    except Exception as e:
        print(f'ERROR parsing XYZ: {e}', file=sys.stderr)
        sys.exit(3)

    save_path = None
    if args.save is not None:
        if args.save is True:
            save_path = os.path.join(os.path.dirname(xyz_path), 'water_trimer.png')
        else:
            save_path = args.save

    plot_molecule(symbols, coords, save_path=save_path)


if __name__ == '__main__':
    main()
