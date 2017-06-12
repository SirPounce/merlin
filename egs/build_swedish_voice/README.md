


This is a modified version of the Merlin toolkit to enable building a swedish voice when one has access HTS style rich label files.
If you have another style format you will need to replace the question file in merlin/misc/questions

Setup
-----

To setup voice: 

./01_setup.sh give_a_voice_name

Prepare Data
------------

Put wav files in database/wav
Put txt files in database/txt
Put lab files in database/lab

Run

./02_prepare_data.sh

Run Merlin
----------

Once after setup, use below script to create acoustic, duration models and perform final test synthesis:

Commment or uncomment the steps you want to run not that I did not include any automated setup of the test synthesis so far.

./03_run_merlin.sh



