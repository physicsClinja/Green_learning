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
    from mpl_toolkits.mplot3d import Axes3D

    X = coords[:, 0]
    Y = coords[:, 1]
    Z = coords[:, 2]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Separate atom types
    idx_O = [i for i, s in enumerate(symbols) if s.upper().startswith('O')]
    idx_H = [i for i, s in enumerate(symbols) if s.upper().startswith('H')]

    # Draw bonds first so atom markers appear on top
    # Robust rule: draw a single bond for each pair (i<j) only when one is O and the other is H
    # and their distance is within a reasonable cutoff for an O-H bond. This avoids O-O lines
    # and prevents double-plotting (which can make lines look darker).
    bond_cutoff = 1.3  # angstrom, generous cutoff for O-H
    n_atoms = len(symbols)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            si = symbols[i].upper()
            sj = symbols[j].upper()
            # Only consider O-H pairs
            if (si.startswith('O') and sj.startswith('H')) or (si.startswith('H') and sj.startswith('O')):
                di = coords[i]
                dj = coords[j]
                dist = np.linalg.norm(di - dj)
                if dist <= bond_cutoff and dist > 0.0:
                    # plot single bond once
                    ax.plot([di[0], dj[0]], [di[1], dj[1]], [di[2], dj[2]], color='lightgray', linewidth=1.0, zorder=1)

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
