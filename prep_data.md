1. Create `blobfuse_connection.cfg` file with following content

```
accountName <DATASTORE ACCOUNT NAME>
accountKey <DATASTORE ACCOUNT KEY>
containerName <CONTAINER NAME>
```

2. Create empty directory for mounting

```
mkdir blobstore
```

3. Mount datastore

```
sudo blobfuse blobstore --tmp-path=/mnt/resource/blobfusetmp  --config-file=blobfuse_connection.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120 -o allow_other
```

4. Add kaggle credentials

```
export KAGGLE_USERNAME=xxxxxxxxx
export KAGGLE_KEY=xxxxxxxxxxxxxx
```

5. Download kaggle dataset into mounted blobstore

```
mkdir blobstore/cars
cd blobstore/cars
kaggle datasets download -d jutrera/stanford-car-dataset-by-classes-folder
unzip stanford-car-dataset-by-classes-folder.zip
rm stanford-car-dataset-by-classes-folder.zip
```

6. Refactor data 

```
mv car_data car_data_1
mv car_data/car_data car_data
rm -rf car_data_1
```

7. Unmount blobstore

```
cd ../..
sudo umount blobfuse
```

7. Register `stanford-cars' folder as a dataset from Studio UI