#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <file1> [file2 ...] <target_dir>"
    exit 1
fi

target_dir="${@: -1}"
files=("${@:1:$#-1}")

mkdir -p "$target_dir"
target_dir_abs="$(realpath "$target_dir")"

for src in "${files[@]}"; do
    if [ ! -f "$src" ]; then
        echo "Skipping: $src is not a file"
        continue
    fi

    src_abs="$(realpath "$src")"
    filename="$(basename "$src")"
    dest="$target_dir_abs/$filename"

    echo "Copying: $src_abs -> $dest"

    rsync -ah --info=progress2 "$src_abs" "$dest"

    echo "Creating symlink at original location..."

    backup="${src_abs}.backup_before_symlink"

    mv "$src_abs" "$backup"

    if ln -s "$dest" "$src_abs"; then
        rm "$backup"
        echo "Done: $src_abs -> $dest"
    else
        echo "Symlink failed, restoring original"
        mv "$backup" "$src_abs"
        exit 1
    fi
done
