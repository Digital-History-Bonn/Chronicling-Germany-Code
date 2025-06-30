DATA_PATH = $1
ENV_NAME=${2:-'pipeline'}
FTP_ARGS=$3

echo ls -f -1 | sftp FTP_ARGS > "${DATA_PATH}file_list.txt"

eval "$(conda shell.bash hook)"
conda activate $2
python script/large_data_pipeline/clean_file_list.py -d $DATA_PATH