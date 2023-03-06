<table align="center"><tr><td align="center" width="9999">
<img src="https://nicolasfeyer.ch/logo-square.png" align="center" width="150" alt="Project icon">

<br/><br/>

Python code to perform keyword spotting using SIFT features as showed by Rusiñol and al. on their paper [[1]](#1).

</td></tr></table>

## Description

This project proposes a keywords spotting system using SIFT descriptors. It uses a dense sampling method of SIFT descriptors. It groups the descriptors using the K-means algorithm to generate visual words. Then, the visual words are grouped into patches of definable size and a histogram of their frequency is produced (BoVW). The visual words of each patch are then weighted with TF-IDF and transformed with LSA.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Authors](#authors)
- [References](#references)

## Installation

This project has the following dependecies.

```
Package                Version
---------------------- -----------
beautifulsoup4         4.11.1
configparser           5.3.0
matplotlib             3.6.0
numpy                  1.23.3
opencv-python          4.6.0.66
psutil                 5.9.2
python                 3.10.6
scikit-image           0.19.3
scikit-learn           1.1.2
seaborn                0.12.1
Shapely                1.8.5.post1
svgpath2mpl            1.0.0
tabulate               0.8.10
tqdm                   4.64.1
```
In addition to the above dependencies, we need to add the ```cyvlfeat``` library which is a wrapper for the C++ [VLFeat](https://www.vlfeat.org/) library.

Here are the steps to install the project within a conda virtual environment.

```console
git clone https://github.com/nicolasfeyer/SIFT-KWS.git
cd SIFT-KWS
conda create --name sift-kws python=3.10
conda activate sift-kws
conda install -c conda-forge cyvlfeat
pip install -r requirements.context
```

## Usage

The script contains 2 actions :
- ```generate``` : Used to generate the corpus
- ```query``` : Used to query the corpus

### Corpus generation
The ```generate``` action has the following parameters:

| Short | Long                   | Description                                                                                                              | Required |
|-------|------------------------|--------------------------------------------------------------------------------------------------------------------------|----------|
| -h    | --help                 | Show the help message and exit                                                                                           | No       |
| -cn   | --corpus-name          | The name of the corpus to generate                                                                                       | Yes      |
| -i    | --images_folder        | The path of the image folders                                                                                            | Yes      |
| -pt   | --part-corpus          | Images index to be used to build the corpus. By default, all the image in the 'images-folder' are used. First index is 0 | No       |
| -ss   | --sift-step            | The SIFT sampling step, i.e. the distance between SIFT descriptors during their extraction                               | Yes      |
| -bs   | --bin-sizes            | The sizes of one of the 16 spatial bin of the SIFT descriptor in pixels                                                  | Yes      |
| -mt   | --magnitude-thresholds | The magnitude limit below which a SIFT descriptor is discarded                                                           | Yes      |
| -k    | -codebook-size         | Size of the codebook, i.e. the number of clusters in which the SIFT descriptors are grouped                              | Yes      |
| -h    | --patch-height         | The height of the patches                                                                                                | Yes      |
| -ws   | --patch-widths         | The widths of the patch                                                                                                  | Yes      |
| -ps   | --patch_sampling       | The sampling step of the patch                                                                                           | Yes      |
| -t    | --topics               | The number of topics used for the LSA transformation. It must be equal or lower than the codebook size K                 | Yes      |


### Querying
The ```query``` action has the following parameters:

| Short | Long                  | Description                                                                                                        | Required |
|-------|-----------------------|--------------------------------------------------------------------------------------------------------------------|----------|
| -h    | --help                | Show the help message and exit                                                                                     | No       |
| -cn   | --corpus-name         | The name of the corpus to be used                                                                                  | Yes      |
| -te   | --templates-folder    | The path of the templates folder                                                                                   | Yes      |
| -st   | --strategy            | The strategy to use for the querying. It can be only-from-corpus, only-from-non-corpus, intersection, union        | No       |
| -tt   | --template-text-limit | If specified, only the templates containing more than the specified number of characters are used for the querying | No       |
| -gt   | -ground-truth-folder  | The path of the ground truth folder. If not specified, no evaluation is performed                                  | No       |

#### Strategies

The querying process can be configured with 4 different strategies. This parameter is tied with the parameter ```part-corpus``` which indicates which images were used to build the corpus:

- ```only-from-corpus``` : Only the templates extracted from the image used to build the corpus will be used as queries
- ```only-from-non-corpus``` : Only the templates extracted from the image **not** used to build the corpus will be used as queries
- ```intersection``` : Use only the template extracted from the image **not** used to build the corpus representing words that are represented in the corpus
- ```union``` : Use all templates available

### Custom ground truth parser

You can write you own ground truth parser OR adapt you own ground truth format to fit the following template in the form of a TSV.

```template-filename  page_no word  x y width height```

If you want to write your own parser, it has to inheritate the class ```scripts/gt_parser/ground_truth_parser```

### Examples

```console
python kws-sift.py generate -cn "washington" -i dataset/washington -ss 5 -bs 5 10 20 -mt 5 5 5 -k 512 -H 80 -ws 80 160 240 320 -t 512 -ps 27 -pt 1-10
```

```console
python kws-sift.py query -cn "washington" -te dataset/washington/templates -gt dataset/washington/gt -st "only-from-corpus"
```

## Datasets

Here are the adapted to KWS datasets:

- George Washington Letters (GW20) ([download](https://drive.google.com/file/d/1v1j4whEwmUdO_yauwLL_qCS2GV_T7y5n/view))
- Pinkas Dataset (PK)[[2]](#2) ([download](https://drive.google.com/file/d/1fYvzBTisD6XmUQJmqawg7sGeNJd5K9NO/view))

## Authors

Nicolas Feyer

## References
<a id="1">[1]</a>
Marçal Rusiñol, David Aldavert, Ricardo Toledo, Josep Lladós,
Efficient segmentation-free keyword spotting in historical document collections,
Pattern Recognition,
Volume 48, Issue 2,
2015,
Pages 545-555,
ISSN 0031-3203,
https://doi.org/10.1016/j.patcog.2014.08.021.
(https://www.sciencedirect.com/science/article/pii/S0031320314003355)
</br>
<a id="2">[2]</a>
Kurar Barakat, B., El-Sana, J., & Rabaev, I. (2019). The Pinkas Dataset. 2019 International Conference on Document Analysis and Recognition (ICDAR), 732–737. https://doi.org/10.1109/ICDAR.2019.00122
