#!/bin/bash
#usage ./movefiles.sh experiments/voicename 

#rm $1/duration_model/data/label_phone_align/*
#rm $1/duration_model/data/file_id_list_full.scp
#rm $1/acoustic_model/data/label_phone_align/*
#rm $1/acoustic_model/datafile_id_list_full.scp

mkdir -p $1/duration_model/data/label_phone_align/
mkdir -p $1/acoustic_model/data/label_phone_align/ 
mkdir -p $1/test_synthesis/
cp database/lab/* $1/duration_model/data/label_phone_align/
cp database/lab/* $1/acoustic_model/data/label_phone_align/
 
./scripts/makefileidlist.sh  $1/duration_model/data/label_phone_align/ $1/duration_model/data/file_id_list_full.scp
cp $1/duration_model/data/file_id_list_full.scp $1/acoustic_model/data/file_id_list_full.scp
#cp $1/acoustic_model/data/file_id_list_full.scp $1/test_synthesis/test_file_id_list.scp
