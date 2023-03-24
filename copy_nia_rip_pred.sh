date
mkdir ../scinet_nia
mkdir ../scinet_nia/datasets
mkdir ../scinet_nia/datasets/NIA
mkdir ../scinet_nia/datasets/NIA/obs_qc_100p
cp -r datasets/NIA/meta_csv ../scinet_nia/datasets/NIA/
echo 'directory setting done'

cp -r data_process ../scinet_nia/
echo 'preprocessing file copy done'

cp -r datasets/NIA/obs_qc_100p/* ../scinet_nia/datasets/NIA/obs_qc_100p/
echo 'test data copy done'

cp -r datasets/NIA/*scaler_csvBase.pkl ../scinet_nia/datasets/NIA/
cp -r datasets/NIA/*test_allYear_csvBase.pkl ../scinet_nia/datasets/NIA/
echo 'test meta data copy done'

cp -r exp ../scinet_nia/
cp -r experiments ../scinet_nia/
cp -r metrics ../scinet_nia/
cp -r models ../scinet_nia/
cp -r utils ../scinet_nia/
echo 'mmdet module copy done'

conda create -p /home/sdh/scinet_nia/venv --clone scinet
echo 'virtual environment copy done'

cp  dockerfile ../scinet_nia/dockerfile
echo 'dockerfile copy done'

cp run_NIA.py ../scinet_nia/
echo 'NIA run file copy done'

tar -zcf ../scinet_nia.tar.gz ../scinet_nia/
echo 'package compression done'

# rm -r ../scinet_nia
# echo 'removing files done'
date

