mkdir -p ZSTL_Data

#move the relevant file to run data_preprocessing script

cp -r ./src ./Data_Proccessing
cp ./ZSTL_GPU.py ./Data_Proccessing

#AwA2
curl -SL http://cvml.ist.ac.at/AwA2/AwA2-features.zip | tar -xf - -C ./ZSTL_Data
curl -SL https://cvml.ist.ac.at/AwA2/AwA2-base.zip | tar -xf - -C ./ZSTL_Data/Animals_with_Attributes2

cp -r ./ZSTL_Data/Animals_with_Attributes2/Animals_with_Attributes2/ ./ZSTL_Data/Animals_with_Attributes2
rm -r ./ZSTL_Data/Animals_with_Attributes2/Animals_with_Attributes2
mkdir -p ./ZSTL_Data/Animals_with_Attributes2/splitedTask

#LastFM
mkdir -p ./ZSTL_Data/hetrec2011-lastfm-2k
curl -SL http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip | tar -xf - -C ./ZSTL_Data/hetrec2011-lastfm-2k
mkdir -p ./ZSTL_Data/hetrec2011-lastfm-2k/extracted_feature