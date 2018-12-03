# image-augmentor

## Image Scraper

https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en

## Instructions

Install python 3.x (developed on python 3.7)

Run in command prompt

`pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely imgaug`

Place the `augment.py` in your image root folder.

Open command prompt and change directory to your image root folder `cd /Path/To/Root/Image/Folder`

Run in command prompt

`python augment.py`

Augmented files will be added to a new folder in the root image folder under the name `augmented`. This folder will contain matching subdirs based on extracted subdirs from root image folder. Running multiple times will generate different images, but will overrite current augmented files. So be sure to save the augmented images somewhere else before rerunning.
