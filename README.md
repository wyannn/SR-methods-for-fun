# super-resolution
A collection of super-resolution models & algorithms

Detail introduction of each model is in corresponding sub-folds.

## Requirement
- python3.6
- numpy
- tensorflow 1.8.0

## Models
- [VDSR](https://github.com/icpm/super-resolution/tree/master/VDSR)
- [EDSR](https://github.com/icpm/super-resolution/tree/master/EDSR)
- [DCRN](https://github.com/icpm/super-resolution/tree/master/DRCN)
- [SubPixelCNN](https://github.com/icpm/super-resolution/tree/master/SubPixelCNN)
- [SRCNN](https://github.com/icpm/super-resolution/tree/master/SRCNN)
- [FSRCNN](https://github.com/icpm/super-resolution/tree/master/FSRCNN)
- [SRGAN](https://github.com/icpm/super-resolution/tree/master/SRGAN)
- [DBPN](https://github.com/icpm/super-resolution/tree/master/DBPN)

## Usage
train:

```bash
$ python3 train.py 
```

super resolve:

```bash
$ python3 test.py
```