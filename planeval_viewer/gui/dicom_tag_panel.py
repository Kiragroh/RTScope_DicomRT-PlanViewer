from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class DicomTagEntry:
    label: str
    paths: tuple[Path, ...]
    dataset: Dataset | None = None


class DicomTagPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[Path] = []
        self.entries: list[DicomTagEntry] = []
        self.dataset: Dataset | None = None
        self.pending_folder: Path | None = None

        self.file_combo = QComboBox()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Tag, Keyword oder Value suchen")
        self.expand_button = QPushButton("Alles auf")
        self.save_copy_button = QPushButton("Save copy")
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Tag", "Keyword", "VR", "Value"])
        self.tree.setAlternatingRowColors(True)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(self.file_combo, 2)
        controls.addWidget(self.search_edit, 2)
        controls.addWidget(self.expand_button)
        controls.addWidget(self.save_copy_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(controls)
        layout.addWidget(self.tree, 1)

        self.file_combo.currentIndexChanged.connect(self._load_selected_file)
        self.search_edit.textChanged.connect(self._apply_search)
        self.expand_button.clicked.connect(self.tree.expandAll)
        self.save_copy_button.clicked.connect(self._save_copy)
        self.tree.itemChanged.connect(self._item_changed)

    def set_folder(self, folder: Path) -> None:
        self.pending_folder = None
        self.entries = _dicom_entries(folder)
        self.paths = [entry.paths[0] for entry in self.entries if entry.paths]
        self.file_combo.blockSignals(True)
        try:
            self.file_combo.clear()
            for entry in self.entries:
                self.file_combo.addItem(entry.label, str(entry.paths[0]) if entry.paths else "")
        finally:
            self.file_combo.blockSignals(False)
        if self.entries:
            self.file_combo.setCurrentIndex(0)
            self._load_entry(self.entries[0])
        else:
            self.dataset = None
            self.tree.clear()

    def set_pending_folder(self, folder: Path) -> None:
        self.pending_folder = folder
        self.paths = []
        self.entries = []
        self.dataset = None
        self.file_combo.blockSignals(True)
        try:
            self.file_combo.clear()
            self.file_combo.addItem("DICOM tags werden beim Oeffnen dieses Tabs geladen", "")
        finally:
            self.file_combo.blockSignals(False)
        self.tree.clear()

    def _load_selected_file(self, index: int) -> None:
        if 0 <= index < len(self.entries):
            self._load_entry(self.entries[index])

    def _load_entry(self, entry: DicomTagEntry) -> None:
        if entry.dataset is not None:
            self.dataset = entry.dataset
            self._populate_dataset(self.dataset)
            return
        if entry.paths:
            self._load_path(entry.paths[0])

    def _load_path(self, path: Path) -> None:
        try:
            self.dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            self.dataset = None
        self._populate_dataset(self.dataset)

    def _populate_dataset(self, dataset: Dataset | None) -> None:
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            if dataset is not None:
                _add_dataset_items(self.tree.invisibleRootItem(), dataset)
                self.tree.resizeColumnToContents(0)
                self.tree.resizeColumnToContents(1)
        finally:
            self.tree.blockSignals(False)
        self._apply_search(self.search_edit.text())

    def _apply_search(self, text: str) -> None:
        needle = text.strip().lower()
        for index in range(self.tree.topLevelItemCount()):
            _filter_item(self.tree.topLevelItem(index), needle)

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 3:
            return
        element = item.data(3, Qt.ItemDataRole.UserRole)
        if element is None or getattr(element, "VR", "") == "SQ":
            return
        try:
            element.value = item.text(3)
        except Exception:
            return

    def _save_copy(self) -> None:
        if self.dataset is None:
            return
        filename, _selected = QFileDialog.getSaveFileName(
            self,
            "DICOM copy speichern",
            "dicom_copy.dcm",
            "DICOM (*.dcm);;All files (*)",
        )
        if filename:
            self.dataset.save_as(filename)


def _dicom_entries(folder: Path) -> list[DicomTagEntry]:
    headers: list[tuple[Path, Dataset]] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if getattr(ds, "SOPClassUID", None) or getattr(ds, "Modality", None):
            headers.append((path, ds))

    ct_groups: dict[str, list[tuple[Path, Dataset]]] = {}
    entries: list[DicomTagEntry] = []
    for path, ds in headers:
        modality = str(getattr(ds, "Modality", "") or "").upper()
        if modality == "CT":
            series_uid = str(getattr(ds, "SeriesInstanceUID", "") or path.parent)
            ct_groups.setdefault(series_uid, []).append((path, ds))
            continue
        entries.append(DicomTagEntry(_file_label_from_dataset(path, ds), (path,)))

    for series_uid, items in ct_groups.items():
        items.sort(key=_ct_slice_sort_key)
        dataset = _ct_series_dataset(series_uid, items)
        description = str(getattr(items[0][1], "SeriesDescription", "") or series_uid)
        entries.append(
            DicomTagEntry(
                f"CT series - {description} ({len(items)} slices)",
                tuple(path for path, _ds in items),
                dataset,
            )
        )
    return sorted(entries, key=lambda item: item.label.lower())


def _dicom_paths(folder: Path) -> list[Path]:
    return [entry.paths[0] for entry in _dicom_entries(folder) if entry.paths]


def _file_label(path: Path) -> str:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return path.name
    return _file_label_from_dataset(path, ds)


def _file_label_from_dataset(path: Path, ds: Dataset) -> str:
    modality = str(getattr(ds, "Modality", "") or "DICOM")
    return f"{modality} - {path.name}"


def _ct_slice_sort_key(item: tuple[Path, Dataset]) -> tuple[float, int, str]:
    path, ds = item
    position = getattr(ds, "ImagePositionPatient", None)
    z = 0.0
    if position is not None and len(position) >= 3:
        try:
            z = float(position[2])
        except (TypeError, ValueError):
            z = 0.0
    instance = int(getattr(ds, "InstanceNumber", 0) or 0)
    return (z, instance, path.name.lower())


def _ct_series_dataset(series_uid: str, items: list[tuple[Path, Dataset]]) -> Dataset:
    first = items[0][1]
    summary = Dataset()
    summary.Modality = "CT"
    summary.SeriesInstanceUID = series_uid
    summary.SeriesDescription = str(getattr(first, "SeriesDescription", "") or "")
    summary.NumberOfSeriesRelatedInstances = len(items)
    for keyword in (
        "StudyInstanceUID",
        "FrameOfReferenceUID",
        "Rows",
        "Columns",
        "PixelSpacing",
        "ImageOrientationPatient",
        "SliceThickness",
    ):
        if hasattr(first, keyword):
            setattr(summary, keyword, getattr(first, keyword))
    referenced = []
    for path, ds in items:
        item = Dataset()
        if hasattr(ds, "SOPClassUID"):
            item.ReferencedSOPClassUID = ds.SOPClassUID
        if hasattr(ds, "SOPInstanceUID"):
            item.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        if hasattr(ds, "InstanceNumber"):
            item.InstanceNumber = ds.InstanceNumber
        if hasattr(ds, "ImagePositionPatient"):
            item.ImagePositionPatient = ds.ImagePositionPatient
        item.ReferencedFileID = _dicom_file_id(path)
        referenced.append(item)
    summary.ReferencedImageSequence = Sequence(referenced)
    return summary


def _dicom_file_id(path: Path) -> str:
    file_id = "".join(char if char.isalnum() else "_" for char in path.stem.upper()).strip("_")
    return (file_id or "CT")[:16]


def _add_dataset_items(parent: QTreeWidgetItem, dataset: Dataset) -> None:
    for element in dataset:
        tag = f"({element.tag.group:04X},{element.tag.element:04X})"
        item = QTreeWidgetItem(
            [
                tag,
                element.keyword or element.name,
                element.VR,
                _element_value_text(element.value),
            ]
        )
        item.setData(3, Qt.ItemDataRole.UserRole, element)
        if element.VR != "SQ":
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        parent.addChild(item)
        if element.VR == "SQ" and isinstance(element.value, Sequence):
            for index, child_dataset in enumerate(element.value):
                child = QTreeWidgetItem(["", f"Item {index + 1}", "", ""])
                item.addChild(child)
                _add_dataset_items(child, child_dataset)


def _element_value_text(value: object) -> str:
    if isinstance(value, Sequence):
        return f"{len(value)} item(s)"
    text = str(value)
    if len(text) > 500:
        return text[:500] + "..."
    return text


def _filter_item(item: QTreeWidgetItem, needle: str) -> bool:
    own_match = not needle or any(needle in item.text(column).lower() for column in range(4))
    child_match = False
    for index in range(item.childCount()):
        if _filter_item(item.child(index), needle):
            child_match = True
    visible = own_match or child_match
    item.setHidden(not visible)
    if needle and child_match:
        item.setExpanded(True)
    return visible
