
"""
The "core" of mantipython
Coordinate everything and fit an entire map.
Takes paths to Herschel maps and fits SEDs in parallel.
Created: January 29, 2020
"""
__author__ = "Ramsey Karim"


#### Obviously take these out, we should have arguments for this kind of stuff.
data_dir = "/n/sgraraid/filaments/data/TEST4/pacs70_cal_test/RCW49/processed/1342255009_reproc350/"
data_fns = {
    70: "PACS70um-image-remapped-conv-plus000102.fits", # -plus000102
    160: "PACS160um-image-remapped-conv-plus000343.fits", # -plus000343
    250: "SPIRE250um-image-remapped-conv.fits",
    350: "SPIRE350um-image-remapped-conv.fits",
    # 500: "SPIRE500um-image-remapped-conv.fits",
}
err_fns = {
    70: "PACS70um-error-remapped-conv.fits",
    160: "PACS160um-error-remapped-conv.fits",
    250: "SPIRE250um-error-remapped-conv.fits",
    350: "SPIRE350um-error-remapped-conv.fits",
    # 500: "SPIRE500um-error-remapped-conv.fits",
}


def fit_entire_map():
    # need a better name
    # should probably implement before picking names... chickens before they hatch
    pass
