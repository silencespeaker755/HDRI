# High Dynamic Range Imaging

## 0. Prerequisites

**Installation** (under **python 3.6.9**)
> * numpy >= 1.22.3
> * opencv-python >= 4.5.5.64
> * ExifRead >= 2.3.2
> * matplotlib >= 3.5.1
> * joblib >= 1.1.0


## 1. Quick Start

### HDR Reconstruction

* Debevec's Method
* Robertson's Method

### Tone Mapping

* Photographic

  * Global Operator
  * Local Operator

### Input directory

```
[input_directory]/
.
├── debevec.py
├── robertson.py
├── HDRI_generate.py
├── HDRI_transform.py
├── imageIO.py
├── [Photos]
│   ├── [tiger]
│   └── [shadowbox]
├── Radiance_generate.py
├── [DebevecData]
├── [RobertsonData]
├── toneMapping.py
├── utils.py
├── requirement.txt
└── README.md
```
**Note**: These images in `Photo/` must contain `EXIF` data (especially `EXIF ExposureTime`)

## 2. Execution
### Basic usage
1. install require packages
```powershell
    pip3 install -r requirement.txt
```
2. Generates blue, green and red radiance files in numpy array format.
```powershell
    python3 Radiance_generate.py -s <source_dir> -d <save_dir> -m <method> [-e <epoch>]
```
3. Generates **HDR image** according to blue, green and red .npy files.
```powershell
    python3 HDRI_generate.py -b <blue> -g <green> -r <red> -d <target_dir> [-o <output>]    
```
4. According to different tonemapping methods, generates **LDR image** with blue, green and red .npy files.
```powershell
    python3 HDRI_transform.py -b <blue> -g <green> -r <red> -m <method> -d <target_dir>    [-o <output>]
```

### Aditional options
The following default arguments can be changed according to user's preference
```
Radiance_generate.py:
    -r <radiance> : change .npy files' prefix
    -g <inverse_response_curve>: change inverse CRFs' image prefix
    
HDRI_generate.py:
    -r <radiance> : change .npy files' prefix
    -g <inverse_response_curve>: change inverse CRFs' image prefix

HDRI_transform.py:
    -a <a> : change .npy files' prefix, default=0.7
    -e <epsilon>: change inverse CRFs' image prefix, default=0.05
    -s <scale>: change inverse CRFs' image prefix, default=15
    -p <phi>: change inverse CRFs' image prefix, default=10
```
## 3. Simple Example
#### Debevec Implementation

```powershell
sh sh/debevec.sh
```
or
```powershell
# every argument is already set default value for debevec
python3 Radiance_generate.py
python3 HDRI_generate.py
python3 HDRI_transform.py
```

#### Robertson Implementation

```powershell
sh sh/robertson.sh
```
or
```powershell
python3 Radiance_generate.py -s Photos/shadowbox -d RobertsonData -m robertson
python3 HDRI_generate.py -b RobertsonData/Radiance_B.npy -g RobertsonData/Radiance_G.npy -r RobertsonData/Radiance_R.npy -d RobertsonData
python3 HDRI_transform.py -b RobertsonData/Radiance_B.npy -g RobertsonData/Radiance_G.npy -r RobertsonData/Radiance_R.npy -m global -d RobertsonData
```
