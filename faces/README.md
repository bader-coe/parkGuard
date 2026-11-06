# Face dataset (private)

This folder is intentionally **git-ignored** because it contains biometric
images. Keep your own photos here; they are never uploaded by this project.

## Layout

Create one subfolder per enrolled person. The folder name is the identity key
used by the SQLite `permits.car_owner` column:

```text
faces/
├── alice/
│   ├── photo-1.jpg
│   └── photo-2.jpg
└── bob/
    └── photo-1.jpg
```

Requirements:

- Use 2–5 clear, front-facing photos per person.
- The face must be visible and reasonably large in each image.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
- The folder name must exactly match the owner name you pass to
  `add_user_db.py` (for example `--owner alice`).

After adding photos, regenerate the local face encodings:

```bash
python train_faces.py
python add_user_db.py --owner alice --permit 1001 \
  --student-id 30000001 --display-name "Alice Example"
```

`main.py` also enrolls images from this folder automatically on first run when
the SQLite encoding table is empty. To force a full rebuild, run:

```bash
python main.py --rebuild-encodings
```
