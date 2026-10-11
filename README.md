# PanoStichPy

Stitching af DJI Air 3-panoramaer fra de originale enkeltbilleder:

| Type | Billeder | Resultat |
|---|---|---|
| 360° kugle | 33 | 360×180 equirectangular med GPano-metadata og udfyldt himmel |
| 180° panorama | 21 | ca. 270×100°, beskåret equirectangular med GPano "cropped area" |
| Vidvinkel | 9 | ca. 130×110°, beskåret equirectangular med GPano "cropped area" |
| Lodret | 3 | højt billede i transvers projektion (kan gå helt ned under dronen) |

Typen genkendes automatisk ud fra billedernes vinkler (eller vælges i programmet). Dronen tager første
billede i den retning/hældning piloten peger, og de øvrige rundt om det.

## Setup
```
python -m pip install -r requirements.txt
```

## Program (anbefalet)
Dobbeltklik på **`PanoStitch.bat`** (eller `python app/main.py`).

1. Vælg en panorama-mappe – eller hele `PANORAMA`-mappen fra SD-kortet, så findes alle panoramaer.
   Man kan også trække en mappe ind i vinduet.
2. Vælg kvalitet og lys. Standardindstillingerne er gode.
3. Tryk **Stitch**. Resultatet gemmes som `<dato>_<mappe>_360.jpg` (default i `Billeder\PanoStitch`) med
   360-metadata, så Facebook, Google Photos m.fl. viser det som 360°.
4. Se resultatet i den indbyggede 360-viewer (træk for at se rundt, scroll for zoom).

Mangler der billeder (fx hvis dronen stoppede før tid), bliver de områder sorte.

## Kommandolinje
```
python step2_align.py input/001_0174 --out output/step2
python step3_blend.py input/001_0174 --cameras output/step2/cameras.json --out output/step3
```
Resultat: `output/step3/step3_pano.jpg` (8192×4096).

## Struktur
```
input/001_0174/          originalbilleder fra DJI (PANORAMA-mappen på SD-kortet)
panostitch/
  metadata.py            læser DJI XMP (gimbal-vinkler, GPS) + EXIF (brændvidde, eksponering)
  camera.py              pinhole-kamera, rotationer, parallakse-model (flad jord)
  projection.py          equirectangular rays, remap-tabeller, feather-vægte
  features.py            SIFT + matching af nabobilleder (metadata-filter + RANSAC)
  bundle.py              bundle adjustment: rotation pr. billede, fælles f/k1/k2, evt. kamerapositioner
  blend.py               gain-udligning, graph-cut-sømme, multi-band blending (wrap-around ved ±180°)
  zenith.py              udfyldning af hullet i himlen over dronen
  xmp.py                 GPano 360-metadata i JPEG
  layout.py              panoramatyper, valg af lærred (projektion/udsnit), beskæring
  pipeline.py            hele forløbet som biblioteksfunktioner (fremdrift + annullering) – bruges af GUI og scripts
  imageio.py             billed-I/O der virker med æ/ø/å i stier
  render.py              enkle previews (feather, sømme, dækning)
app/
  main.py                start af programmet, farvetema
  main_window.py         hovedvindue
  viewer.py              interaktiv 360-viewer
  worker.py              kører stitching i baggrunden
PanoStitch.bat           starter programmet uden konsolvindue
step1_metadata_only.py   trin 1: placering kun ud fra DJI-metadata
step2_align.py           trin 2: matching + bundle adjustment -> cameras.json
step3_blend.py           trin 3: eksponering, sømme, blending -> step3_pano.jpg
tools/diagnose.py        GPS-drift, fejl pr. billedpar, fejlkort over panoramaet
tools/compare_crops.py   udsnit af samme område fra flere billeder
output/                  resultater
```

## Koordinatsystem
- Verden: x = øst, y = op, z = nord. Yaw er kompasretning (0 = nord, +90 = øst), pitch positiv opad.
- Kamera (OpenCV): x = højre, y = ned, z = frem.
- DJI's gimbal-vinkler anvendes i rækkefølgen yaw → roll → pitch; positiv roll = højre side ned
  (verificeret på vidvinkel-panoramaet, hvis sidekolonner har roll ±9,9°).
- 360°: midten af billedet = startretningen (første billede), eller `--center-yaw`. Kompasretningen gemmes i
  `GPano:PoseHeadingDegrees`.
- Delvise panoramaer: lærredet centreres på det dækkede område (`projection.Canvas`); lodrette panoramaer
  bruger en transvers ramme, så projektionens poler ligger til siderne i stedet for lige under dronen.

## Trin
1. **Kun metadata** – placerer billederne ud fra DJI's gimbal-vinkler. Fejl på 1–3°.
2. **Justering** – retter yaw, pitch og roll pr. billede ud fra billedindholdet.
   - DJI's `GimbalRollDegree` er altid 0, men de reelle roll-fejl er op til ~3°.
   - `--parallax fit` (standard): dronen flytter sig op til ~1 m under optagelsen. Kamerapositionerne
     estimeres (med GPS som udgangspunkt), og jorden antages flad `RelativeAltitude` under dronen.
     Halverer fejlen på jorden tæt på dronen. `--parallax off` slår det fra.
   - Billeder uden brugbare matches (ren himmel) følger deres naboer. Får et billede en urealistisk
     rettelse (> `--max-correction`, typisk pga. matches på skyer der flytter sig), ignoreres dets matches.
3. **Blending**
   - Gain-udligning i lineært lys; `--anchor ground` (standard) bevarer jordbilledernes DJI-eksponering.
   - Graph-cut-sømme beregnet i lav opløsning (`--seam-width`), lagt uden om biler/bygninger.
   - Multi-band blending (`--levels`, standard 9) som wrapper rundt ved ±180°.
4. **Afslutning** – 360°: hullet over dronen udfyldes med glat himmel (kun hvis det er det normale hul; mangler
   der billeder, efterlades området sort). Delvise panoramaer beskæres til det største fuldt dækkede rektangel
   (`--no-crop` beholder de sorte kanter). Billedet gemmes med GPano-metadata, undtagen lodrette panoramaer.
