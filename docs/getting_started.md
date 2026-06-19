# Getting Started

RTScope PlanEval Viewer is research software for local DICOM-RT review. It is
not validated for clinical treatment decisions.

## 1. Start the App

Download the latest Windows build from the
[GitHub Releases page](https://github.com/Kiragroh/RTScope_DicomRT-PlanViewer/releases/latest).
Extract the ZIP file and start `RTScopePlanEvalViewer.exe`.

Windows may show a SmartScreen warning because the executable is not code-signed.

## 2. Prepare a Case Folder

Use one local folder per anonymized case. Subfolders are allowed.

Recommended contents:

- one CT image series from a single `SeriesInstanceUID`
- one RTSTRUCT referencing the CT frame of reference
- one RTDOSE aligned to the plan or CT geometry
- optional one or more RTPLAN files

If a complete `DoseSummationType=PLAN` RTDOSE is present, RTScope uses it. If
only compatible beam or field doses are present, they are summed on the fly.

Do not use folders that contain unrelated patients or mixed CT series. Do not
commit or upload DICOM files, screenshots with identifiable anatomy, DICOM UIDs,
names, MRNs or accession numbers.

## 3. Load the Case

Click `Open` and select the case folder. The app scans subfolders, loads the CT,
RTSTRUCT, RTDOSE and any RTPLAN files, then pre-renders the main views.

If the folder contains multiple RTPLAN files, RTScope first asks which plan
variant should be loaded. Later, use `Plan waehlen` in the top toolbar to open
the same selection window again without reloading the whole case. The compact
`Plan` selector can also switch quickly between loaded variants. This is
intended for plan variants that share the same image/structure context.

The app also opens CT/RTSTRUCT/RTDOSE cases without an RTPLAN. Plan-specific
features such as MLC/BEV playback are only available when an RTPLAN is present.

## 4. Check Structures and Mapping

Use the left `ROIs` panel to control visibility in 2D, 3D and DVH views.

- `Alle an` shows all structures.
- `Alle aus` hides all structures.
- `Nur Matches` shows structures with a RefDB/Hub match.
- `Mapping` lets you manually map a local ROI to a reference name.

Manual mappings are stored locally and can be exported from the `Mappings` tab.

## 5. Review Images, Dose and Isodoses

Use `Axial`, `Sagittal` and `Coronal` to switch 2D views. Use the slice slider,
the vertical scrollbar or the mouse wheel to navigate. Hold `Ctrl` while
scrolling to zoom.

Use the dose scale next to the image to set the displayed dose range. Switch the
mode from `Overlay` to `Isodosen` to show prescription-relative isodose lines.
The 100% reference uses the prescription dose when available.

The persistent 3D panel can show surface or volume CT rendering, dose surfaces,
PTV/OAR meshes and the isocenter. Use `3D einklappen` when you need a larger 2D
view.

## 6. Compute QA

Click `Compute QA`. Then use the lower right tabs:

- `DVH / Constraints` for DVH statistics and constraint checks.
- `Plan` for monitor units, plan geometry and PAM complexity.
- `MLC / BEV` for beam/control-point aperture review and playback.

The target selector affects target-dependent metrics such as CI, GI, HI and PAM.
Change it manually if automatic target detection is wrong.

## 7. RefDB / Hub and Offline Mode

If a RefDB/Hub service is available, configure it before starting the app:

```powershell
$env:PLANEVAL_REFDB_URLS = "http://hub-host:5001;http://fallback-host:5001"
```

Without a Hub connection, the app uses the local cache and synthetic offline
example tables. These examples are for demos and tests only.

## 8. Privacy Screenshots

Use the `CT Blur` toolbar toggle before taking screenshots. It blurs the CT
texture while keeping dose, isodoses and contours visible.
