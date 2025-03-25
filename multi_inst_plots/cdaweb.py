import os
import sunpy
import cdflib

from sunpy.net import Fido
from sunpy.net import attrs as a

def cdaweb_download_fido(dataset, startdate, enddate, path=None, max_conn=5):
    """
    Downloads dataset files via SunPy/Fido from CDAWeb

    Parameters
    ----------
    dataset : {str}
        Name of dataset:
        - 'PSP_FLD_L3_RFS_HFR'
        - 'PSP_FLD_L3_RFS_LFR'
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or
        "standard" datetime string (e.g., "2021/04/15") (enddate must always be
        later than startdate)
    path : {str}, optional
        Local path for storing downloaded data, by default None
    max_conn : {int}, optional
        The number of parallel download slots used by Fido.fetch, by default 5

    Returns
    -------
    List of downloaded files
    """
    trange = a.Time(startdate, enddate)
    cda_dataset = a.cdaweb.Dataset(dataset)
    try:
        result = Fido.search(trange, cda_dataset)
        filelist = [i[0].split('/')[-1] for i in result.show('URL')[0]]
        filelist.sort()
        if path is None:
            filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
        elif type(path) is str:
            filelist = [path + os.sep + f for f in filelist]
        downloaded_files = filelist

        # Check if file with same name already exists in path
        for i, f in enumerate(filelist):
            if os.path.exists(f) and os.path.getsize(f) == 0:
                os.remove(f)
            if not os.path.exists(f):
                downloaded_file = Fido.fetch(result[0][i], path=path, max_conn=max_conn)
    except (RuntimeError, IndexError):
        print(f'Unable to obtain "{dataset}" data for {startdate}-{enddate}!')
        downloaded_files = []
    return downloaded_files

if __name__ == "__main__":
    files = cdaweb_download_fido("PSP_FLD_L3_RFS_HFR", "2023/02/01", "2023/02/02")
    cdf = cdflib.CDF(files[0])
    print(cdf.cdf_info())

