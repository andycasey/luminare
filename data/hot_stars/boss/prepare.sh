mkdir lm0001 lm0002 lm0003 lm0004 lm0005 lm0006 lm0007 lm0008 lp0000 lp0001 lp0002 lp0003 lp0004 lp0005 lp0006 lp0007 lp0008
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lm0001_spectra.tar.gz -C lm0001
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lm0002_spectra.tar.gz -C lm0002
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lm0003_spectra.tar.gz -C lm0003
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lm0004_spectra.tar.gz -C lm0004
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lm0005_spectra.tar.gz -C lm0005
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lm0006_spectra.tar.gz -C lm0006
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lm0007_spectra.tar.gz -C lm0007
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lm0008_spectra.tar.gz -C lm0008
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0000_spectra.tar.gz -C lp0000
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0001_spectra.tar.gz -C lp0001
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0002_spectra.tar.gz -C lp0002
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0003_spectra.tar.gz -C lp0003
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0004_spectra.tar.gz -C lp0004
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0005_spectra.tar.gz -C lp0005
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0006_spectra.tar.gz -C lp0006
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0007_spectra.tar.gz -C lp0007
tar -xzvf /uufs/chpc.utah.edu/common/home/u6027908/AstraHotBOSS/BOSS_lp0008_spectra.tar.gz -C lp0008

python prepare.py
