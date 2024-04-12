wget https://snap.stanford.edu/data/twitch.zip
unzip twitch.zip -d data/
rm twitch.zip
cd data
mv twitch linkteller-data
cd linkteller-data/DE
mv musae_DE.json musae_DE_features.json
cd ..
rm *.txt
