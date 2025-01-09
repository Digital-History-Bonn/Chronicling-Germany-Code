DATA_PATH=$1

cd $DATA_PATH
mkdir xml_results
mv images/page/* xml_results
rm images/*.jpg
rm -rf images/page