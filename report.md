# Details of our team's current work

> until newest update in Jun 22, 2025

## Datasets

|Code Type|Set Size(train/val + test)|Reference|
|:-:|:-:|:-|
|**ArUco**(cls: 0)|2997 + 0|:one:[FlyingArUco v2](https://zenodo.org/records/14053985)|
|**1D Barcode**(cls: 1)|5614 + 428|:one:[Brcode-qrcode-kuihuama Repo](https://github.com/simplew2011/barcode_qrcode_kuihuama)<br/>:two:WeChat Proprietary Dataset(private)|
|**Kuihua Code**(cls: 2)|7194 + 203|same as barcode|
|**QR Code**(cls: 3)|5792 + 1134|same as barcode|

> For the dataset of `Brcode-qrcode-kuihuama Repo`: It's a combination of multiple 1D barcode/QR code datasets collected by [simplew2011](https://github.com/simplew2011), and one of its reference is [BenSouchet/barcode-datasets](https://github.com/BenSouchet/barcode-datasets) which include many popular 1D Barcode dataset(e.g. `InventBar`, `ParcelBar`, `SBD`, `Muenster BarcodeDB`)
>
> We only selected part of them to apply in our training and testing.



## Source

- We find most of our image data from:
  - [Roboflow](https://roboflow.com/)
  - [Kaggle](https://www.kaggle.com/)
  - [Zenodo](https://zenodo.org/)



## Development Guide

### Dataset form

- we originate dataset as:

```
# [image_path] [width] [height] [cls]
[src_X] [src_Y] [dst_X] [dst_Y] [kps1_X] [kps1_Y] [kps1_state] [kps2_X] [kps2_Y] [kps2_state] [kps3_X] [kps3_Y] [kps3_state]
```

- since we deprecate key points, the last 9 data were set to -1
- following modification should be done:
  - [ ] delete `kps` data
  - [ ] make `cls` target-specified other than image-specified
- we hope finally our dataset like this:

```
# [image_path] [width] [height]
[src_X] [src_Y] [dst_X] [dst_Y] [cls] /* one target */
[src_X] [src_Y] [dst_X] [dst_Y] [cls] /* if exists multi-target */
```

