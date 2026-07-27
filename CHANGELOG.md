# Changelog

## 0.1.2 - Build 9 (2026-07-27)

- Direkt startbare Standalone-EXE im STR-Archiv
- `RTScopePlanEvalViewer.exe` liegt direkt im gemeinsamen STR-Standalone-Ordner und kann dort ohne Python gestartet werden.
- Das Portable-ZIP bleibt zusaetzlich im Unterordner `release` verfuegbar.
- Die README beschreibt den direkten Start und der Hub verwendet weiterhin den gemeinsamen Ordner als Git-Pfad.

## 0.1.2 - Build 8 (2026-07-27)

- Gemeinsame Standalone-Ablage im STR-Archiv
- Eine gemeinsame, patientendatenfreie Standalone-Kopie liegt im STR-Archiv und ist fuer berechtigte UKL-Arbeitsplaetze erreichbar.
- Der STR-Hub verwendet diese Ablage als primaeren Git-Pfad fuer die Versionsanzeige.
- Die README beschreibt den gemeinsamen Pfad und den separaten GitHub-Sourcecode.

## 0.1.2 - Build 7 (2026-07-27)

- Mehrplan-Zuordnung, Ladeoverlay und schlanker Windows-Portable-Build
- RTPLAN- und RTSTRUCT-Referenzen werden bei mehreren Planvarianten sicher bidirektional zugeordnet.
- Mehrdeutige DICOM-Dateien werden nicht mehr automatisch mehreren Planen zugewiesen.
- Der Ladeoverlay und der gemeinsame OpenGL-Kontext reduzieren den sichtbaren Ladeblitz.
- Der Windows-Portable-Build schliesst unbenutzte ML- und GPU-Abhaengigkeiten aus.
