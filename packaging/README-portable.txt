PanoStitch {version} - panoramas from DJI drones (portable Python version)
https://github.com/Montago/PanoStitchPy
======================================================================

Stitches the single images from the drone's panorama modes into one image:
  360° sphere (33 images), 180° panorama (21), wide angle (9) and vertical (3).
360°, 180° and wide-angle results carry 360° metadata, so they can be shown in
Facebook, Google Photos, Kuula, VR headsets etc. Developed and tested with the DJI Air 3.

This folder runs on an installed Python. For a version that needs no installation at all,
download PanoStitch-<version>-win64.zip from the GitHub releases page.


FIRST TIME
----------
1. Python 3.11 or newer must be installed (https://www.python.org/downloads/).
   Tick "Add python.exe to PATH" during the installation.
2. Double-click Installer.bat. It installs the packages the program uses
   (OpenCV, NumPy, SciPy, Pillow, PySide6 - about 300 MB, needs internet).


HOW TO USE
----------
1. Double-click PanoStitch.bat.
2. Choose the folder with a panorama - or the whole PANORAMA folder from the drone's SD card
   to find all panoramas. You can also drag a folder onto the window.
   Use the ORIGINAL single images (e.g. PANORAMA\001_0174\PANO_0001.JPG ...).
3. Choose type, quality and exposure and press "Stitch". The type is detected automatically
   and shown next to each panorama in the list.
4. The result is saved as <date>_<folder>_<type>.jpg in the "Save to" folder
   (default: Pictures\PanoStitch) and shown in the built-in 360° viewer.
5. Try the views "Tiny planet" and "Inverted" above the viewer: drag sideways to turn, up/down to tilt,
   scroll to zoom. "Save view..." saves what is inside the frame as a high-resolution image.

Time per 360° panorama (33 images): Fast approx. 3 min, Standard approx. 4 min, High approx. 10 min.


GOOD TO KNOW
------------
- The program creates no working folders and never writes to the image folder / SD card.
  The only things saved are the finished panorama, saved views (and your settings in Windows).
- Partial panoramas (180°, wide angle, vertical) are cropped so there are no black borders.
- If images are missing (e.g. the drone stopped early), those areas are left black.
- The drone's camera cannot point straight up; for 360° the gap in the sky is filled with smooth sky.
- "Correct for the drone drifting" makes the ground close to the drone sharper when it drifts in the wind.
- "Exposure": "Keep the drone's exposure on the ground" gives a bright landscape but a larger burnt-out sun;
  "Balanced" keeps more colour in the sky but makes the ground a little darker.


LICENSE
-------
PanoStitch is MIT licensed (LICENSE.txt).
