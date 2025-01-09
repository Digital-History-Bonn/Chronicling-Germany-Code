DATA_PATH = $1

cd DATA_PATH
mkdir images
mv get_images.sh images/get_images.sh
cd images
bash get_images.sh
rename .jpeg .jpg *.jpeg