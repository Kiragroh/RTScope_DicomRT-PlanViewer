# Changelog

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
