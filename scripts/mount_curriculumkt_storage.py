"""Mount the authorized external project at its frozen canonical path."""
import os
import subprocess
from currkt_storage import ROOT, IMAGE, STATE, read, volume_info


if __name__ == '__main__':
    plan = read(STATE / 'plan.json')
    if ROOT.is_mount():
        assert volume_info(ROOT)['VolumeUUID'] == plan['image_volume_uuid']
        print('CurriculumKT external storage is already mounted.')
    else:
        assert IMAGE.exists()
        ROOT.mkdir(exist_ok=True)
        assert not any(ROOT.iterdir()), 'Refusing to mount over local project files'
        subprocess.run(['/usr/bin/hdiutil','attach',str(IMAGE),'-mountpoint',str(ROOT),'-nobrowse','-noautoopen'],check=True)
        assert volume_info(ROOT)['VolumeUUID'] == plan['image_volume_uuid']
        print('Mounted CurriculumKT external storage at',ROOT)
