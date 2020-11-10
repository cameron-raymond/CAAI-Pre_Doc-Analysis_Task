# CAAI Applied Machine Learning Pre-Doc - Analysis Task

---

- **Name:** Cameron Raymond
- **Current Date:** 
- **Due Date:** November 20, 2020
<!-- - **Submission link:** https://forms.gle/tagRjeGCQBHVZNiR9 -->

## 1. Find the The IMDB-WIKI dataset**

IMDB-Wiki dataset can be found [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) and was created by Rothe, Timofte & Van Gool in 2015. It's the largest publically available dataset of human faces with gender, age, and name. It contains more than 500 thousand+ images with the associated meta information:
* `dob`: date of birth (Matlab serial date number)
* `photo_taken`: year when the photo was taken
* `full_path`: path to file
* `gender`: 0 for female and 1 for male, NaN if unknown
* `name`: name of the celebrity
* `face_location`: location of the face. 
* `face_score`: detector score (the higher the better). Inf implies that no face was found in the image and the face_location then just returns the entire image
* `second_face_score`: detector score of the face with the second highest score. This is useful to ignore images with more than one face. second_face_score is NaN if no second face was detected.
* `celeb_names` (IMDB only): list of all celebrity names
* `celeb_id` (IMDB only): index of celebrity name

The original data set is quite unweildy, but open-sourced code from [`imdeepmind`](https://github.com/imdeepmind/processed-imdb-wiki-dataset) provides a set of python scripts for processing the data. The processed metadata is stored in this project's root directory in the file `FILE.csv`. Since