Git is available in the working directory:
  Merlin version:  Merlin_V0-17-gcb339f9
  branch:  master
Architecture: x86_64
Distribution: Ubuntu 14.04.5 LTS
HOSTNAME=fant
USER=joniva
 
PATH:
    /usr/local/sbin
    /usr/local/bin
    /usr/sbin
    /usr/bin
    /sbin
    /bin
    /usr/games
LD_LIBRARY_PATH:
PYTHONPATH:
PYTHONBIN: python
MERLIN_THEANO_FLAGS:
    cuda.root=/usr/local/cuda-8.0
    floatX=float32
    on_unused_input=ignore
 
Python version: version 2.7.6 (default, Jun 22 2015, 17:58:13) [GCC 4.8.2]
OK
 
Python Numpy version: version 1.11.2
OK
 
Python Theano version: version 0.9.0dev3.dev-25f0dee338b901070e021d431c938102890bc69f
OK
 
2017-05-03 17:47:21,244     INFO           test: Testing Merlin classes
2017-05-03 17:47:21,244     INFO           test: Build models without training
2017-05-03 17:47:21,465     INFO           test:   DeepRecurrentNetwork ['TANH']
2017-05-03 17:47:21,492     INFO           test:     OK
2017-05-03 17:47:21,492     INFO           test:   DeepRecurrentNetwork ['TANH', 'TANH']
2017-05-03 17:47:21,506     INFO           test:     OK
2017-05-03 17:47:21,506     INFO           test:   DeepRecurrentNetwork ['LSTM', 'LSTM']
2017-05-03 17:47:21,764     INFO           test:     OK
2017-05-03 17:47:21,764     INFO           test:   DeepRecurrentNetwork ['SLSTM', 'SLSTM']
2017-05-03 17:47:21,844     INFO           test:     OK
There are differences against master-Merlin_V0-17-gcb339f9
(you're testing local changes, differences can be expected)
 
Characterization test on short data (../egs/slt_arctic/s1)...
Step 1: setting up experiments directory and the training data files...
data is ready!
Merlin default voice settings configured in conf/global_settings.cfg
setup done...!
Duration configuration settings stored in conf/duration_slt_arctic_demo.conf
Acoustic configuration settings stored in conf/acoustic_slt_arctic_demo.conf
Duration configuration settings stored in conf/test_dur_synth_slt_arctic_demo.conf
Acoustic configuration settings stored in conf/test_synth_slt_arctic_demo.conf
Step 2: training duration model...
Architecture: x86_64
Distribution: Ubuntu 14.04.5 LTS
HOSTNAME=fant
USER=joniva
 
PATH:
    /usr/local/sbin
    /usr/local/bin
    /usr/sbin
    /usr/bin
    /sbin
    /bin
    /usr/games
LD_LIBRARY_PATH:
PYTHONPATH:
PYTHONBIN: python
MERLIN_THEANO_FLAGS:
    cuda.root=/usr/local/cuda-8.0
    floatX=float32
    on_unused_input=ignore
 
Running on GPU id=0 ...
Using gpu device 0: GeForce GTX 1080 (CNMeM is disabled, cuDNN 5105)
2017-05-03 17:47:24,053    DEBUG  configuration: successfully read and parsed user configuration file /data/joniva/software/merlin/egs/slt_arctic/s1/conf/duration_slt_arctic_demo.conf
2017-05-03 17:47:24,053     INFO  configuration:           Paths:work has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model
2017-05-03 17:47:24,053     INFO  configuration:           Paths:data has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data
2017-05-03 17:47:24,053     INFO  configuration:           Paths:plot has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/plots
2017-05-03 17:47:24,053     INFO  configuration:         Utility:plot has default value False
2017-05-03 17:47:24,053     INFO  configuration:      Utility:profile has default value False
2017-05-03 17:47:24,054     INFO  configuration:   Paths:file_id_list has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/file_id_list_demo.scp
2017-05-03 17:47:24,054     INFO  configuration:   Paths:test_id_list has    user value None
2017-05-03 17:47:24,054     INFO  configuration:         Paths:GV_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/GV
2017-05-03 17:47:24,054     INFO  configuration:   Paths:in_stepw_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/stepw
2017-05-03 17:47:24,054     INFO  configuration:     Paths:in_mgc_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/mgc
2017-05-03 17:47:24,054     INFO  configuration:     Paths:in_lf0_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/lf0
2017-05-03 17:47:24,054     INFO  configuration:     Paths:in_bap_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/bap
2017-05-03 17:47:24,054     INFO  configuration:      Paths:in_sp_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/sp
2017-05-03 17:47:24,054     INFO  configuration:  Paths:in_seglf0_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/lf03
2017-05-03 17:47:24,054     INFO  configuration:      Paths:in_F0_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/F0
2017-05-03 17:47:24,054     INFO  configuration:    Paths:in_Gain_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/Gain
2017-05-03 17:47:24,054     INFO  configuration:     Paths:in_HNR_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/HNR
2017-05-03 17:47:24,054     INFO  configuration:     Paths:in_LSF_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/LSF
2017-05-03 17:47:24,055     INFO  configuration: Paths:in_LSFsource_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/LSFsource
2017-05-03 17:47:24,055     INFO  configuration: Paths:in_seq_dur_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/S2S_dur
2017-05-03 17:47:24,055     INFO  configuration:     Paths:in_dur_dir has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/dur
2017-05-03 17:47:24,055     INFO  configuration: Paths:nn_norm_temp_dir has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/step_hidden9
2017-05-03 17:47:24,055     INFO  configuration: Labels:process_labels_in_work_dir has default value False
2017-05-03 17:47:24,055     INFO  configuration:   Labels:label_style has default value HTS
2017-05-03 17:47:24,055     INFO  configuration:    Labels:label_type has    user value state_align
2017-05-03 17:47:24,055     INFO  configuration:   Labels:label_align has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align
2017-05-03 17:47:24,055     INFO  configuration: Labels:question_file_name has    user value /data/joniva/software/merlin/misc/questions/questions-radio_dnn_416.hed
2017-05-03 17:47:24,055     INFO  configuration: Labels:silence_pattern has    user value ['*-sil+*']
2017-05-03 17:47:24,055     INFO  configuration: Labels:subphone_feats has    user value none
2017-05-03 17:47:24,055     INFO  configuration: Labels:additional_features has default value {}
2017-05-03 17:47:24,056     INFO  configuration: Labels:xpath_file_name has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/xml_labels/xpaths.txt
2017-05-03 17:47:24,056     INFO  configuration:  Labels:label_config has default value configuration/examplelabelconfigfile.py
2017-05-03 17:47:24,056     INFO  configuration: Labels:add_frame_features has    user value False
2017-05-03 17:47:24,056     INFO  configuration: Labels:fill_missing_values has default value False
2017-05-03 17:47:24,056     INFO  configuration: Labels:xpath_label_align has default value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align
2017-05-03 17:47:24,056     INFO  configuration: Labels:enforce_silence has default value False
2017-05-03 17:47:24,056     INFO  configuration: Labels:remove_silence_using_binary_labels has default value False
2017-05-03 17:47:24,056     INFO  configuration: Labels:precompile_xpaths has default value True
2017-05-03 17:47:24,056     INFO  configuration: Labels:iterate_over_frames has default value True
2017-05-03 17:47:24,056     INFO  configuration: Labels:appended_input_dim has default value 0
2017-05-03 17:47:24,056     INFO  configuration:     Data:buffer_size has    user value 200000
2017-05-03 17:47:24,056     INFO  configuration: Data:train_file_number has    user value 50
2017-05-03 17:47:24,056     INFO  configuration: Data:valid_file_number has    user value 5
2017-05-03 17:47:24,056     INFO  configuration: Data:test_file_number has    user value 5
2017-05-03 17:47:24,057     INFO  configuration:       Paths:log_path has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/log
2017-05-03 17:47:24,057     INFO  configuration:       Paths:log_file has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/log/mylogfilename.log
2017-05-03 17:47:24,057     INFO  configuration: Paths:log_config_file has    user value /data/joniva/software/merlin/egs/slt_arctic/s1/conf/logging_config.conf
2017-05-03 17:47:24,057     INFO  configuration:           Paths:sptk has default value tools/bin/SPTK-3.9
2017-05-03 17:47:24,057     INFO  configuration:       Paths:straight has default value tools/bin/straight
2017-05-03 17:47:24,057     INFO  configuration:          Paths:world has default value tools/bin/WORLD
2017-05-03 17:47:24,057     INFO  configuration: Architecture:network_type has default value RNN
2017-05-03 17:47:24,057     INFO  configuration: Architecture:model_type has default value DNN
2017-05-03 17:47:24,057     INFO  configuration: Architecture:hidden_layer_type has    user value ['TANH', 'TANH', 'TANH', 'TANH']
2017-05-03 17:47:24,057     INFO  configuration: Architecture:output_layer_type has default value LINEAR
2017-05-03 17:47:24,057     INFO  configuration: Architecture:sequential_training has    user value False
2017-05-03 17:47:24,057     INFO  configuration: Architecture:dropout_rate has    user value 0.0
2017-05-03 17:47:24,057     INFO  configuration:  Architecture:scheme has default value stagewise
2017-05-03 17:47:24,058     INFO  configuration: Architecture:index_to_project has default value 0
2017-05-03 17:47:24,058     INFO  configuration: Architecture:projection_insize has default value 10000
2017-05-03 17:47:24,058     INFO  configuration: Architecture:projection_outsize has default value 10
2017-05-03 17:47:24,058     INFO  configuration: Architecture:initial_projection_distrib has default value gaussian
2017-05-03 17:47:24,058     INFO  configuration: Architecture:projection_weights_output_dir has default value some_path
2017-05-03 17:47:24,058     INFO  configuration: Architecture:layers_with_projection_input has default value [0]
2017-05-03 17:47:24,058     INFO  configuration: Architecture:projection_learning_rate_scaling has default value 1.0
2017-05-03 17:47:24,058     INFO  configuration: Architecture:learning_rate has    user value 0.002
2017-05-03 17:47:24,058     INFO  configuration: Architecture:L2_regularization has default value 1e-05
2017-05-03 17:47:24,058     INFO  configuration: Architecture:L1_regularization has default value 0.0
2017-05-03 17:47:24,058     INFO  configuration: Architecture:batch_size has    user value 64
2017-05-03 17:47:24,058     INFO  configuration: Architecture:training_epochs has    user value 25
2017-05-03 17:47:24,058     INFO  configuration: Architecture:hidden_activation has default value tanh
2017-05-03 17:47:24,058     INFO  configuration: Architecture:output_activation has    user value linear
2017-05-03 17:47:24,059     INFO  configuration: Architecture:hidden_layer_size has    user value [512, 512, 512, 512]
2017-05-03 17:47:24,059     INFO  configuration: Architecture:private_hidden_sizes has default value [1024]
2017-05-03 17:47:24,059     INFO  configuration: Architecture:stream_weights has default value [1.0]
2017-05-03 17:47:24,059     INFO  configuration: Architecture:private_l2_reg has default value 1e-05
2017-05-03 17:47:24,059     INFO  configuration: Architecture:warmup_epoch has    user value 10
2017-05-03 17:47:24,059     INFO  configuration: Architecture:warmup_momentum has    user value 0.3
2017-05-03 17:47:24,059     INFO  configuration: Architecture:momentum has default value 0.9
2017-05-03 17:47:24,059     INFO  configuration: Architecture:warmup_epoch has    user value 10
2017-05-03 17:47:24,059     INFO  configuration: Architecture:mdn_component has default value 1
2017-05-03 17:47:24,059     INFO  configuration: Architecture:var_floor has default value 0.01
2017-05-03 17:47:24,059     INFO  configuration: Architecture:beta_opt has default value False
2017-05-03 17:47:24,059     INFO  configuration: Architecture:eff_sample_size has default value 0.8
2017-05-03 17:47:24,059     INFO  configuration: Architecture:mean_log_det has default value -100.0
2017-05-03 17:47:24,060     INFO  configuration: Architecture:start_from_trained_model has default value _
2017-05-03 17:47:24,060     INFO  configuration: Architecture:use_rprop has default value 0
2017-05-03 17:47:24,060     INFO  configuration:          Outputs:mgc has default value 60
2017-05-03 17:47:24,060     INFO  configuration:         Outputs:dmgc has default value 180
2017-05-03 17:47:24,060     INFO  configuration:          Outputs:vuv has default value 1
2017-05-03 17:47:24,060     INFO  configuration:          Outputs:lf0 has default value 1
2017-05-03 17:47:24,060     INFO  configuration:         Outputs:dlf0 has default value 3
2017-05-03 17:47:24,060     INFO  configuration:          Outputs:bap has default value 25
2017-05-03 17:47:24,060     INFO  configuration:         Outputs:dbap has default value 75
2017-05-03 17:47:24,060     INFO  configuration:          Outputs:cmp has default value 259
2017-05-03 17:47:24,060     INFO  configuration:    Outputs:stepw_dim has default value 55
2017-05-03 17:47:24,060     INFO  configuration:  Outputs:temp_sp_dim has default value 1025
2017-05-03 17:47:24,060     INFO  configuration:   Outputs:seglf0_dim has default value 7
2017-05-03 17:47:24,061     INFO  configuration:    Outputs:delta_win has default value [-0.5, 0.0, 0.5]
2017-05-03 17:47:24,061     INFO  configuration:      Outputs:acc_win has default value [1.0, -2.0, 1.0]
2017-05-03 17:47:24,061     INFO  configuration:      Outputs:do_MLPG has default value True
2017-05-03 17:47:24,061     INFO  configuration:           Outputs:F0 has default value 1
2017-05-03 17:47:24,061     INFO  configuration:          Outputs:dF0 has default value 3
2017-05-03 17:47:24,061     INFO  configuration:         Outputs:Gain has default value 1
2017-05-03 17:47:24,061     INFO  configuration:        Outputs:dGain has default value 3
2017-05-03 17:47:24,061     INFO  configuration:          Outputs:HNR has default value 5
2017-05-03 17:47:24,061     INFO  configuration:         Outputs:dHNR has default value 15
2017-05-03 17:47:24,061     INFO  configuration:          Outputs:LSF has default value 30
2017-05-03 17:47:24,061     INFO  configuration:         Outputs:dLSF has default value 90
2017-05-03 17:47:24,061     INFO  configuration:    Outputs:LSFsource has default value 10
2017-05-03 17:47:24,061     INFO  configuration:   Outputs:dLSFsource has default value 30
2017-05-03 17:47:24,061     INFO  configuration:      Outputs:seq_dur has default value 1
2017-05-03 17:47:24,062     INFO  configuration: Outputs:remove_silence_from_dur has default value True
2017-05-03 17:47:24,062     INFO  configuration:          Outputs:dur has    user value 5
2017-05-03 17:47:24,062     INFO  configuration: Outputs:dur_feature_type has default value numerical
2017-05-03 17:47:24,062     INFO  configuration: Outputs:output_feature_normalisation has default value MVN
2017-05-03 17:47:24,062     INFO  configuration: Streams:multistream_switch has default value False
2017-05-03 17:47:24,062     INFO  configuration: Streams:output_features has    user value ['dur']
2017-05-03 17:47:24,062     INFO  configuration: Streams:gen_wav_features has default value ['mgc', 'bap', 'lf0']
2017-05-03 17:47:24,062     INFO  configuration: Waveform:vocoder_type has default value STRAIGHT
2017-05-03 17:47:24,062     INFO  configuration:  Waveform:samplerate has default value 48000
2017-05-03 17:47:24,062     INFO  configuration: Waveform:framelength has default value 4096
2017-05-03 17:47:24,062     INFO  configuration:  Waveform:frameshift has default value 5
2017-05-03 17:47:24,062     INFO  configuration:      Waveform:sp_dim has default value 2049
2017-05-03 17:47:24,062     INFO  configuration:    Waveform:fw_alpha has default value 0.77
2017-05-03 17:47:24,062     INFO  configuration: Waveform:postfilter_coef has default value 1.4
2017-05-03 17:47:24,063     INFO  configuration: Waveform:minimum_phase_order has default value 2047
2017-05-03 17:47:24,063     INFO  configuration:  Waveform:use_cep_ap has default value True
2017-05-03 17:47:24,063     INFO  configuration: Waveform:do_post_filtering has default value True
2017-05-03 17:47:24,063     INFO  configuration:    Waveform:apply_GV has default value False
2017-05-03 17:47:24,063     INFO  configuration: Waveform:test_synth_dir has    user value None
2017-05-03 17:47:24,063     INFO  configuration: Processes:DurationModel has    user value True
2017-05-03 17:47:24,063     INFO  configuration: Processes:AcousticModel has default value False
2017-05-03 17:47:24,063     INFO  configuration: Processes:GenTestList has    user value False
2017-05-03 17:47:24,063     INFO  configuration:    Processes:NORMLAB has    user value True
2017-05-03 17:47:24,063     INFO  configuration:    Processes:MAKEDUR has    user value True
2017-05-03 17:47:24,063     INFO  configuration:    Processes:MAKECMP has    user value True
2017-05-03 17:47:24,063     INFO  configuration:    Processes:NORMCMP has    user value True
2017-05-03 17:47:24,063     INFO  configuration:   Processes:TRAINDNN has    user value True
2017-05-03 17:47:24,063     INFO  configuration:     Processes:DNNGEN has    user value True
2017-05-03 17:47:24,063     INFO  configuration:     Processes:GENWAV has default value False
2017-05-03 17:47:24,064     INFO  configuration:     Processes:CALMCD has    user value True
2017-05-03 17:47:24,064     INFO  configuration:   Processes:NORMSTEP has default value False
2017-05-03 17:47:24,064     INFO  configuration:   Processes:GENBNFEA has default value False
2017-05-03 17:47:24,064     INFO  configuration:   Extensions:mgc_ext has default value .mgc
2017-05-03 17:47:24,064     INFO  configuration:   Extensions:bap_ext has default value .bap
2017-05-03 17:47:24,064     INFO  configuration:   Extensions:lf0_ext has default value .lf0
2017-05-03 17:47:24,064     INFO  configuration:   Extensions:cmp_ext has default value .cmp
2017-05-03 17:47:24,064     INFO  configuration:   Extensions:lab_ext has default value .lab
2017-05-03 17:47:24,064     INFO  configuration:   Extensions:utt_ext has default value .utt
2017-05-03 17:47:24,064     INFO  configuration: Extensions:stepw_ext has default value .stepw
2017-05-03 17:47:24,064     INFO  configuration:    Extensions:sp_ext has default value .sp
2017-05-03 17:47:24,064     INFO  configuration:    Extensions:F0_ext has default value .F0
2017-05-03 17:47:24,064     INFO  configuration:  Extensions:Gain_ext has default value .Gain
2017-05-03 17:47:24,064     INFO  configuration:   Extensions:HNR_ext has default value .HNR
2017-05-03 17:47:24,064     INFO  configuration:   Extensions:LSF_ext has default value .LSF
2017-05-03 17:47:24,065     INFO  configuration: Extensions:LSFsource_ext has default value .LSFsource
2017-05-03 17:47:24,065     INFO  configuration:   Extensions:dur_ext has default value .dur
2017-05-03 17:47:24,065    DEBUG  configuration: loading logging configuration from /data/joniva/software/merlin/egs/slt_arctic/s1/conf/logging_config.conf
2017-05-03 17:47:24,067 [1;32mINFO    [0m  configuration: logging is now fully configured
2017-05-03 17:47:24,067 [1;34mDEBUG   [0m  configuration: setting up output features
2017-05-03 17:47:24,067 [1;34mDEBUG   [0m  configuration:  dur
2017-05-03 17:47:24,067 [1;32mINFO    [0m  configuration:   in_dimension: 5
2017-05-03 17:47:24,067 [1;32mINFO    [0m  configuration:   out_dimension : 5
2017-05-03 17:47:24,067 [1;32mINFO    [0m  configuration:   in_directory : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/dur
2017-05-03 17:47:24,067 [1;32mINFO    [0m  configuration: multistream dimensions: [5]
2017-05-03 17:47:24,067 [1;32mINFO    [0m  configuration: use_rprop : 0
2017-05-03 17:47:24,067 [1;32mINFO    [0m  configuration: dropout_rate : 0.0
2017-05-03 17:47:24,067 [1;32mINFO    [0m  configuration: projection_outsize : 10
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: early_stop_epochs : 5
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: warmup_epoch : 10
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: learning_rate : 0.002
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: layers_with_projection_input : [0]
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: batch_size : 64
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: model_type : DNN
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: hidden_layer_size : [512, 512, 512, 512]
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: projection_learning_rate_scaling : 1.0
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: pretraining_epochs : 10
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: initial_projection_distrib : gaussian
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: index_to_project : 0
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: l2_reg : 1e-05
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: warmup_momentum : 0.3
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: training_epochs : 25
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: hidden_activation : tanh
2017-05-03 17:47:24,068 [1;32mINFO    [0m  configuration: hidden_layer_type : ['TANH', 'TANH', 'TANH', 'TANH']
2017-05-03 17:47:24,069 [1;32mINFO    [0m  configuration: sequential_training : False
2017-05-03 17:47:24,069 [1;32mINFO    [0m  configuration: do_pretraining : False
2017-05-03 17:47:24,069 [1;32mINFO    [0m  configuration: projection_insize : 10000
2017-05-03 17:47:24,069 [1;32mINFO    [0m  configuration: l1_reg : 0.0
2017-05-03 17:47:24,069 [1;32mINFO    [0m  configuration: output_activation : linear
2017-05-03 17:47:24,069 [1;32mINFO    [0m  configuration: pretraining_lr : 0.0001
2017-05-03 17:47:24,069 [1;32mINFO    [0m  configuration: momentum : 0.9
2017-05-03 17:47:24,069 [1;34mDEBUG   [0m  configuration: configuration completed
2017-05-03 17:47:24,069 [1;32mINFO    [0m       main    : Installation information:
2017-05-03 17:47:24,069 [1;32mINFO    [0m       main    :   Merlin directory: /data/joniva/software/merlin
2017-05-03 17:47:24,069 [1;32mINFO    [0m       main    :   PATH:
2017-05-03 17:47:24,069 [1;32mINFO    [0m       main    :       /usr/local/sbin
2017-05-03 17:47:24,069 [1;32mINFO    [0m       main    :       /usr/local/bin
2017-05-03 17:47:24,069 [1;32mINFO    [0m       main    :       /usr/sbin
2017-05-03 17:47:24,069 [1;32mINFO    [0m       main    :       /usr/bin
2017-05-03 17:47:24,069 [1;32mINFO    [0m       main    :       /sbin
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :       /bin
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :       /usr/games
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :   LD_LIBRARY_PATH:
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :   Python version: 2.7.6 (default, Jun 22 2015, 17:58:13) [GCC 4.8.2]
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :     PYTHONPATH:
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :   Numpy version: 1.11.2
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :   Theano version: 0.9.0dev3.dev-25f0dee338b901070e021d431c938102890bc69f
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :     THEANO_FLAGS: mode=FAST_RUN,device=gpu0,cuda.root=/usr/local/cuda-8.0,floatX=float32,on_unused_input=ignore
2017-05-03 17:47:24,070 [1;32mINFO    [0m       main    :     device: gpu0
2017-05-03 17:47:24,080 [1;32mINFO    [0m       main    :   Git is available in the working directory:
2017-05-03 17:47:24,089 [1;32mINFO    [0m       main    :     Merlin version: Merlin_V0-17-gcb339f9
2017-05-03 17:47:24,097 [1;32mINFO    [0m       main    :     branch: master
2017-05-03 17:47:24,105 [1;32mINFO    [0m       main    :     diff to Merlin version:
2017-05-03 17:47:24,105 [1;32mINFO    [0m       main    :       M misc/questions/questions-radio_dnn_416.hed
2017-05-03 17:47:24,105 [1;32mINFO    [0m       main    :       M src/setup_env.sh
2017-05-03 17:47:24,105 [1;32mINFO    [0m       main    :       M test/test_all.sh
2017-05-03 17:47:24,106 [1;32mINFO    [0m       main    :       (all diffs logged in DNN_TANH_TANH_TANH_TANH_LINEAR__dur_50_259_4_512_0.002000_05_47PM_May_03_2017.log.gitdiff)
2017-05-03 17:47:24,115 [1;32mINFO    [0m       main    : Execution information:
2017-05-03 17:47:24,115 [1;32mINFO    [0m       main    :   HOSTNAME: fant
2017-05-03 17:47:24,115 [1;32mINFO    [0m       main    :   USER: joniva
2017-05-03 17:47:24,115 [1;32mINFO    [0m       main    :   PID: 20500
2017-05-03 17:47:24,115 [1;34mDEBUG   [0m       main    : Loaded file id list from /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/file_id_list_demo.scp
2017-05-03 17:47:24,171 [1;34mDEBUG   [0m       labels  : HTS-derived input feature dimension is 1483 + 0 = 1483
2017-05-03 17:47:24,172 [1;32mINFO    [0m       main    : Input label dimension is 1483
2017-05-03 17:47:24,172 [1;32mINFO    [0m       main    : preparing label data (input) using standard HTS style labels
2017-05-03 17:47:24,172 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0001.lab, 185 labels
2017-05-03 17:47:24,255 [1;34mDEBUG   [0m       labels  : made label matrix of 37 frames x 1483 labels
2017-05-03 17:47:24,256 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0002.lab, 210 labels
2017-05-03 17:47:24,351 [1;34mDEBUG   [0m       labels  : made label matrix of 42 frames x 1483 labels
2017-05-03 17:47:24,351 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0003.lab, 205 labels
2017-05-03 17:47:24,444 [1;34mDEBUG   [0m       labels  : made label matrix of 41 frames x 1483 labels
2017-05-03 17:47:24,445 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0004.lab, 140 labels
2017-05-03 17:47:24,507 [1;34mDEBUG   [0m       labels  : made label matrix of 28 frames x 1483 labels
2017-05-03 17:47:24,507 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0005.lab,  85 labels
2017-05-03 17:47:24,545 [1;34mDEBUG   [0m       labels  : made label matrix of 17 frames x 1483 labels
2017-05-03 17:47:24,545 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0006.lab, 165 labels
2017-05-03 17:47:24,620 [1;34mDEBUG   [0m       labels  : made label matrix of 33 frames x 1483 labels
2017-05-03 17:47:24,621 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0007.lab, 200 labels
2017-05-03 17:47:24,712 [1;34mDEBUG   [0m       labels  : made label matrix of 40 frames x 1483 labels
2017-05-03 17:47:24,712 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0008.lab, 120 labels
2017-05-03 17:47:24,766 [1;34mDEBUG   [0m       labels  : made label matrix of 24 frames x 1483 labels
2017-05-03 17:47:24,766 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0009.lab, 200 labels
2017-05-03 17:47:24,856 [1;34mDEBUG   [0m       labels  : made label matrix of 40 frames x 1483 labels
2017-05-03 17:47:24,856 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0010.lab, 205 labels
2017-05-03 17:47:24,950 [1;34mDEBUG   [0m       labels  : made label matrix of 41 frames x 1483 labels
2017-05-03 17:47:24,951 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0011.lab, 170 labels
2017-05-03 17:47:25,028 [1;34mDEBUG   [0m       labels  : made label matrix of 34 frames x 1483 labels
2017-05-03 17:47:25,028 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0012.lab, 180 labels
2017-05-03 17:47:25,109 [1;34mDEBUG   [0m       labels  : made label matrix of 36 frames x 1483 labels
2017-05-03 17:47:25,110 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0013.lab, 260 labels
2017-05-03 17:47:25,228 [1;34mDEBUG   [0m       labels  : made label matrix of 52 frames x 1483 labels
2017-05-03 17:47:25,229 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0014.lab, 180 labels
2017-05-03 17:47:25,310 [1;34mDEBUG   [0m       labels  : made label matrix of 36 frames x 1483 labels
2017-05-03 17:47:25,310 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0015.lab,  95 labels
2017-05-03 17:47:25,353 [1;34mDEBUG   [0m       labels  : made label matrix of 19 frames x 1483 labels
2017-05-03 17:47:25,353 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0016.lab, 205 labels
2017-05-03 17:47:25,445 [1;34mDEBUG   [0m       labels  : made label matrix of 41 frames x 1483 labels
2017-05-03 17:47:25,446 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0017.lab, 275 labels
2017-05-03 17:47:25,571 [1;34mDEBUG   [0m       labels  : made label matrix of 55 frames x 1483 labels
2017-05-03 17:47:25,572 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0018.lab,  75 labels
2017-05-03 17:47:25,605 [1;34mDEBUG   [0m       labels  : made label matrix of 15 frames x 1483 labels
2017-05-03 17:47:25,605 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0019.lab, 220 labels
2017-05-03 17:47:25,705 [1;34mDEBUG   [0m       labels  : made label matrix of 44 frames x 1483 labels
2017-05-03 17:47:25,705 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0020.lab, 195 labels
2017-05-03 17:47:25,793 [1;34mDEBUG   [0m       labels  : made label matrix of 39 frames x 1483 labels
2017-05-03 17:47:25,795 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0021.lab, 160 labels
2017-05-03 17:47:25,868 [1;34mDEBUG   [0m       labels  : made label matrix of 32 frames x 1483 labels
2017-05-03 17:47:25,868 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0022.lab, 270 labels
2017-05-03 17:47:25,992 [1;34mDEBUG   [0m       labels  : made label matrix of 54 frames x 1483 labels
2017-05-03 17:47:25,993 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0023.lab, 360 labels
2017-05-03 17:47:26,161 [1;34mDEBUG   [0m       labels  : made label matrix of 72 frames x 1483 labels
2017-05-03 17:47:26,162 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0024.lab, 245 labels
2017-05-03 17:47:26,274 [1;34mDEBUG   [0m       labels  : made label matrix of 49 frames x 1483 labels
2017-05-03 17:47:26,274 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0025.lab, 190 labels
2017-05-03 17:47:26,360 [1;34mDEBUG   [0m       labels  : made label matrix of 38 frames x 1483 labels
2017-05-03 17:47:26,360 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0026.lab, 185 labels
2017-05-03 17:47:26,444 [1;34mDEBUG   [0m       labels  : made label matrix of 37 frames x 1483 labels
2017-05-03 17:47:26,444 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0027.lab, 240 labels
2017-05-03 17:47:26,556 [1;34mDEBUG   [0m       labels  : made label matrix of 48 frames x 1483 labels
2017-05-03 17:47:26,556 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0028.lab,  85 labels
2017-05-03 17:47:26,593 [1;34mDEBUG   [0m       labels  : made label matrix of 17 frames x 1483 labels
2017-05-03 17:47:26,593 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0029.lab, 200 labels
2017-05-03 17:47:26,684 [1;34mDEBUG   [0m       labels  : made label matrix of 40 frames x 1483 labels
2017-05-03 17:47:26,684 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0030.lab,  70 labels
2017-05-03 17:47:26,715 [1;34mDEBUG   [0m       labels  : made label matrix of 14 frames x 1483 labels
2017-05-03 17:47:26,715 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0031.lab, 130 labels
2017-05-03 17:47:26,773 [1;34mDEBUG   [0m       labels  : made label matrix of 26 frames x 1483 labels
2017-05-03 17:47:26,773 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0032.lab, 240 labels
2017-05-03 17:47:26,883 [1;34mDEBUG   [0m       labels  : made label matrix of 48 frames x 1483 labels
2017-05-03 17:47:26,883 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0033.lab, 240 labels
2017-05-03 17:47:26,992 [1;34mDEBUG   [0m       labels  : made label matrix of 48 frames x 1483 labels
2017-05-03 17:47:26,993 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0034.lab, 220 labels
2017-05-03 17:47:27,092 [1;34mDEBUG   [0m       labels  : made label matrix of 44 frames x 1483 labels
2017-05-03 17:47:27,093 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0035.lab, 235 labels
2017-05-03 17:47:27,199 [1;34mDEBUG   [0m       labels  : made label matrix of 47 frames x 1483 labels
2017-05-03 17:47:27,200 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0036.lab,  95 labels
2017-05-03 17:47:27,242 [1;34mDEBUG   [0m       labels  : made label matrix of 19 frames x 1483 labels
2017-05-03 17:47:27,242 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0037.lab, 140 labels
2017-05-03 17:47:27,305 [1;34mDEBUG   [0m       labels  : made label matrix of 28 frames x 1483 labels
2017-05-03 17:47:27,305 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0038.lab, 105 labels
2017-05-03 17:47:27,352 [1;34mDEBUG   [0m       labels  : made label matrix of 21 frames x 1483 labels
2017-05-03 17:47:27,352 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0039.lab, 155 labels
2017-05-03 17:47:27,422 [1;34mDEBUG   [0m       labels  : made label matrix of 31 frames x 1483 labels
2017-05-03 17:47:27,422 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0040.lab, 155 labels
2017-05-03 17:47:27,493 [1;34mDEBUG   [0m       labels  : made label matrix of 31 frames x 1483 labels
2017-05-03 17:47:27,494 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0041.lab, 125 labels
2017-05-03 17:47:27,550 [1;34mDEBUG   [0m       labels  : made label matrix of 25 frames x 1483 labels
2017-05-03 17:47:27,550 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0042.lab, 170 labels
2017-05-03 17:47:27,627 [1;34mDEBUG   [0m       labels  : made label matrix of 34 frames x 1483 labels
2017-05-03 17:47:27,627 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0043.lab, 230 labels
2017-05-03 17:47:27,732 [1;34mDEBUG   [0m       labels  : made label matrix of 46 frames x 1483 labels
2017-05-03 17:47:27,733 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0044.lab, 195 labels
2017-05-03 17:47:27,821 [1;34mDEBUG   [0m       labels  : made label matrix of 39 frames x 1483 labels
2017-05-03 17:47:27,821 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0045.lab, 150 labels
2017-05-03 17:47:27,889 [1;34mDEBUG   [0m       labels  : made label matrix of 30 frames x 1483 labels
2017-05-03 17:47:27,890 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0046.lab, 175 labels
2017-05-03 17:47:27,968 [1;34mDEBUG   [0m       labels  : made label matrix of 35 frames x 1483 labels
2017-05-03 17:47:27,969 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0047.lab, 195 labels
2017-05-03 17:47:28,056 [1;34mDEBUG   [0m       labels  : made label matrix of 39 frames x 1483 labels
2017-05-03 17:47:28,057 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0048.lab, 150 labels
2017-05-03 17:47:28,124 [1;34mDEBUG   [0m       labels  : made label matrix of 30 frames x 1483 labels
2017-05-03 17:47:28,124 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0049.lab, 180 labels
2017-05-03 17:47:28,206 [1;34mDEBUG   [0m       labels  : made label matrix of 36 frames x 1483 labels
2017-05-03 17:47:28,206 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0050.lab, 240 labels
2017-05-03 17:47:28,316 [1;34mDEBUG   [0m       labels  : made label matrix of 48 frames x 1483 labels
2017-05-03 17:47:28,316 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0051.lab, 235 labels
2017-05-03 17:47:28,424 [1;34mDEBUG   [0m       labels  : made label matrix of 47 frames x 1483 labels
2017-05-03 17:47:28,424 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0052.lab, 125 labels
2017-05-03 17:47:28,481 [1;34mDEBUG   [0m       labels  : made label matrix of 25 frames x 1483 labels
2017-05-03 17:47:28,482 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0053.lab, 205 labels
2017-05-03 17:47:28,575 [1;34mDEBUG   [0m       labels  : made label matrix of 41 frames x 1483 labels
2017-05-03 17:47:28,576 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0054.lab, 100 labels
2017-05-03 17:47:28,620 [1;34mDEBUG   [0m       labels  : made label matrix of 20 frames x 1483 labels
2017-05-03 17:47:28,620 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0055.lab, 250 labels
2017-05-03 17:47:28,733 [1;34mDEBUG   [0m       labels  : made label matrix of 50 frames x 1483 labels
2017-05-03 17:47:28,734 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0056.lab, 165 labels
2017-05-03 17:47:28,808 [1;34mDEBUG   [0m       labels  : made label matrix of 33 frames x 1483 labels
2017-05-03 17:47:28,808 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0057.lab, 115 labels
2017-05-03 17:47:28,859 [1;34mDEBUG   [0m       labels  : made label matrix of 23 frames x 1483 labels
2017-05-03 17:47:28,860 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0058.lab, 235 labels
2017-05-03 17:47:28,968 [1;34mDEBUG   [0m       labels  : made label matrix of 47 frames x 1483 labels
2017-05-03 17:47:28,969 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0059.lab, 140 labels
2017-05-03 17:47:29,032 [1;34mDEBUG   [0m       labels  : made label matrix of 28 frames x 1483 labels
2017-05-03 17:47:29,032 [1;32mINFO    [0m       labels  : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0060.lab, 110 labels
2017-05-03 17:47:29,081 [1;34mDEBUG   [0m       labels  : made label matrix of 22 frames x 1483 labels
2017-05-03 17:47:29,173 [1;34mDEBUG   [0m  acoustic_norm: MinMaxNormalisation created for feature dimension of 1483
2017-05-03 17:47:29,181 [1;32mINFO    [0m  acoustic_norm: across 50 files found min/max values of length 1483:
2017-05-03 17:47:29,182 [1;32mINFO    [0m  acoustic_norm:   min: [[ 0.  0.  0. ...,  0.  0.  0.]]
2017-05-03 17:47:29,182 [1;32mINFO    [0m  acoustic_norm:   max: [[ 0.  0.  0. ...,  0.  0.  0.]]
2017-05-03 17:47:29,226 [1;32mINFO    [0m       main    : saved 1483 vectors to /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_norm_HTS_1483.dat
2017-05-03 17:47:29,226 [1;32mINFO    [0m       main    : creating duration (output) features
2017-05-03 17:47:29,226 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0001.lab, 185 labels
2017-05-03 17:47:29,229 [1;34mDEBUG   [0m       dur     : made duration matrix of 37 frames x 5 features
2017-05-03 17:47:29,229 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0002.lab, 210 labels
2017-05-03 17:47:29,232 [1;34mDEBUG   [0m       dur     : made duration matrix of 42 frames x 5 features
2017-05-03 17:47:29,232 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0003.lab, 205 labels
2017-05-03 17:47:29,235 [1;34mDEBUG   [0m       dur     : made duration matrix of 41 frames x 5 features
2017-05-03 17:47:29,235 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0004.lab, 140 labels
2017-05-03 17:47:29,237 [1;34mDEBUG   [0m       dur     : made duration matrix of 28 frames x 5 features
2017-05-03 17:47:29,237 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0005.lab,  85 labels
2017-05-03 17:47:29,238 [1;34mDEBUG   [0m       dur     : made duration matrix of 17 frames x 5 features
2017-05-03 17:47:29,238 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0006.lab, 165 labels
2017-05-03 17:47:29,240 [1;34mDEBUG   [0m       dur     : made duration matrix of 33 frames x 5 features
2017-05-03 17:47:29,240 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0007.lab, 200 labels
2017-05-03 17:47:29,243 [1;34mDEBUG   [0m       dur     : made duration matrix of 40 frames x 5 features
2017-05-03 17:47:29,243 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0008.lab, 120 labels
2017-05-03 17:47:29,245 [1;34mDEBUG   [0m       dur     : made duration matrix of 24 frames x 5 features
2017-05-03 17:47:29,245 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0009.lab, 200 labels
2017-05-03 17:47:29,247 [1;34mDEBUG   [0m       dur     : made duration matrix of 40 frames x 5 features
2017-05-03 17:47:29,248 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0010.lab, 205 labels
2017-05-03 17:47:29,250 [1;34mDEBUG   [0m       dur     : made duration matrix of 41 frames x 5 features
2017-05-03 17:47:29,250 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0011.lab, 170 labels
2017-05-03 17:47:29,253 [1;34mDEBUG   [0m       dur     : made duration matrix of 34 frames x 5 features
2017-05-03 17:47:29,253 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0012.lab, 180 labels
2017-05-03 17:47:29,255 [1;34mDEBUG   [0m       dur     : made duration matrix of 36 frames x 5 features
2017-05-03 17:47:29,255 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0013.lab, 260 labels
2017-05-03 17:47:29,259 [1;34mDEBUG   [0m       dur     : made duration matrix of 52 frames x 5 features
2017-05-03 17:47:29,259 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0014.lab, 180 labels
2017-05-03 17:47:29,261 [1;34mDEBUG   [0m       dur     : made duration matrix of 36 frames x 5 features
2017-05-03 17:47:29,261 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0015.lab,  95 labels
2017-05-03 17:47:29,262 [1;34mDEBUG   [0m       dur     : made duration matrix of 19 frames x 5 features
2017-05-03 17:47:29,263 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0016.lab, 205 labels
2017-05-03 17:47:29,265 [1;34mDEBUG   [0m       dur     : made duration matrix of 41 frames x 5 features
2017-05-03 17:47:29,265 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0017.lab, 275 labels
2017-05-03 17:47:29,269 [1;34mDEBUG   [0m       dur     : made duration matrix of 55 frames x 5 features
2017-05-03 17:47:29,269 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0018.lab,  75 labels
2017-05-03 17:47:29,270 [1;34mDEBUG   [0m       dur     : made duration matrix of 15 frames x 5 features
2017-05-03 17:47:29,270 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0019.lab, 220 labels
2017-05-03 17:47:29,273 [1;34mDEBUG   [0m       dur     : made duration matrix of 44 frames x 5 features
2017-05-03 17:47:29,273 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0020.lab, 195 labels
2017-05-03 17:47:29,276 [1;34mDEBUG   [0m       dur     : made duration matrix of 39 frames x 5 features
2017-05-03 17:47:29,276 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0021.lab, 160 labels
2017-05-03 17:47:29,278 [1;34mDEBUG   [0m       dur     : made duration matrix of 32 frames x 5 features
2017-05-03 17:47:29,278 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0022.lab, 270 labels
2017-05-03 17:47:29,282 [1;34mDEBUG   [0m       dur     : made duration matrix of 54 frames x 5 features
2017-05-03 17:47:29,282 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0023.lab, 360 labels
2017-05-03 17:47:29,286 [1;34mDEBUG   [0m       dur     : made duration matrix of 72 frames x 5 features
2017-05-03 17:47:29,287 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0024.lab, 245 labels
2017-05-03 17:47:29,290 [1;34mDEBUG   [0m       dur     : made duration matrix of 49 frames x 5 features
2017-05-03 17:47:29,290 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0025.lab, 190 labels
2017-05-03 17:47:29,292 [1;34mDEBUG   [0m       dur     : made duration matrix of 38 frames x 5 features
2017-05-03 17:47:29,292 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0026.lab, 185 labels
2017-05-03 17:47:29,295 [1;34mDEBUG   [0m       dur     : made duration matrix of 37 frames x 5 features
2017-05-03 17:47:29,295 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0027.lab, 240 labels
2017-05-03 17:47:29,298 [1;34mDEBUG   [0m       dur     : made duration matrix of 48 frames x 5 features
2017-05-03 17:47:29,298 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0028.lab,  85 labels
2017-05-03 17:47:29,299 [1;34mDEBUG   [0m       dur     : made duration matrix of 17 frames x 5 features
2017-05-03 17:47:29,299 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0029.lab, 200 labels
2017-05-03 17:47:29,302 [1;34mDEBUG   [0m       dur     : made duration matrix of 40 frames x 5 features
2017-05-03 17:47:29,302 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0030.lab,  70 labels
2017-05-03 17:47:29,303 [1;34mDEBUG   [0m       dur     : made duration matrix of 14 frames x 5 features
2017-05-03 17:47:29,303 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0031.lab, 130 labels
2017-05-03 17:47:29,305 [1;34mDEBUG   [0m       dur     : made duration matrix of 26 frames x 5 features
2017-05-03 17:47:29,305 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0032.lab, 240 labels
2017-05-03 17:47:29,308 [1;34mDEBUG   [0m       dur     : made duration matrix of 48 frames x 5 features
2017-05-03 17:47:29,308 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0033.lab, 240 labels
2017-05-03 17:47:29,311 [1;34mDEBUG   [0m       dur     : made duration matrix of 48 frames x 5 features
2017-05-03 17:47:29,312 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0034.lab, 220 labels
2017-05-03 17:47:29,314 [1;34mDEBUG   [0m       dur     : made duration matrix of 44 frames x 5 features
2017-05-03 17:47:29,315 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0035.lab, 235 labels
2017-05-03 17:47:29,318 [1;34mDEBUG   [0m       dur     : made duration matrix of 47 frames x 5 features
2017-05-03 17:47:29,318 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0036.lab,  95 labels
2017-05-03 17:47:29,319 [1;34mDEBUG   [0m       dur     : made duration matrix of 19 frames x 5 features
2017-05-03 17:47:29,319 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0037.lab, 140 labels
2017-05-03 17:47:29,321 [1;34mDEBUG   [0m       dur     : made duration matrix of 28 frames x 5 features
2017-05-03 17:47:29,321 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0038.lab, 105 labels
2017-05-03 17:47:29,323 [1;34mDEBUG   [0m       dur     : made duration matrix of 21 frames x 5 features
2017-05-03 17:47:29,323 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0039.lab, 155 labels
2017-05-03 17:47:29,325 [1;34mDEBUG   [0m       dur     : made duration matrix of 31 frames x 5 features
2017-05-03 17:47:29,325 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0040.lab, 155 labels
2017-05-03 17:47:29,327 [1;34mDEBUG   [0m       dur     : made duration matrix of 31 frames x 5 features
2017-05-03 17:47:29,327 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0041.lab, 125 labels
2017-05-03 17:47:29,329 [1;34mDEBUG   [0m       dur     : made duration matrix of 25 frames x 5 features
2017-05-03 17:47:29,329 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0042.lab, 170 labels
2017-05-03 17:47:29,331 [1;34mDEBUG   [0m       dur     : made duration matrix of 34 frames x 5 features
2017-05-03 17:47:29,331 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0043.lab, 230 labels
2017-05-03 17:47:29,334 [1;34mDEBUG   [0m       dur     : made duration matrix of 46 frames x 5 features
2017-05-03 17:47:29,334 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0044.lab, 195 labels
2017-05-03 17:47:29,337 [1;34mDEBUG   [0m       dur     : made duration matrix of 39 frames x 5 features
2017-05-03 17:47:29,337 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0045.lab, 150 labels
2017-05-03 17:47:29,339 [1;34mDEBUG   [0m       dur     : made duration matrix of 30 frames x 5 features
2017-05-03 17:47:29,339 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0046.lab, 175 labels
2017-05-03 17:47:29,341 [1;34mDEBUG   [0m       dur     : made duration matrix of 35 frames x 5 features
2017-05-03 17:47:29,341 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0047.lab, 195 labels
2017-05-03 17:47:29,344 [1;34mDEBUG   [0m       dur     : made duration matrix of 39 frames x 5 features
2017-05-03 17:47:29,344 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0048.lab, 150 labels
2017-05-03 17:47:29,346 [1;34mDEBUG   [0m       dur     : made duration matrix of 30 frames x 5 features
2017-05-03 17:47:29,346 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0049.lab, 180 labels
2017-05-03 17:47:29,349 [1;34mDEBUG   [0m       dur     : made duration matrix of 36 frames x 5 features
2017-05-03 17:47:29,349 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0050.lab, 240 labels
2017-05-03 17:47:29,352 [1;34mDEBUG   [0m       dur     : made duration matrix of 48 frames x 5 features
2017-05-03 17:47:29,352 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0051.lab, 235 labels
2017-05-03 17:47:29,355 [1;34mDEBUG   [0m       dur     : made duration matrix of 47 frames x 5 features
2017-05-03 17:47:29,355 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0052.lab, 125 labels
2017-05-03 17:47:29,357 [1;34mDEBUG   [0m       dur     : made duration matrix of 25 frames x 5 features
2017-05-03 17:47:29,357 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0053.lab, 205 labels
2017-05-03 17:47:29,360 [1;34mDEBUG   [0m       dur     : made duration matrix of 41 frames x 5 features
2017-05-03 17:47:29,360 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0054.lab, 100 labels
2017-05-03 17:47:29,361 [1;34mDEBUG   [0m       dur     : made duration matrix of 20 frames x 5 features
2017-05-03 17:47:29,361 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0055.lab, 250 labels
2017-05-03 17:47:29,364 [1;34mDEBUG   [0m       dur     : made duration matrix of 50 frames x 5 features
2017-05-03 17:47:29,365 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0056.lab, 165 labels
2017-05-03 17:47:29,367 [1;34mDEBUG   [0m       dur     : made duration matrix of 33 frames x 5 features
2017-05-03 17:47:29,367 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0057.lab, 115 labels
2017-05-03 17:47:29,368 [1;34mDEBUG   [0m       dur     : made duration matrix of 23 frames x 5 features
2017-05-03 17:47:29,368 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0058.lab, 235 labels
2017-05-03 17:47:29,371 [1;34mDEBUG   [0m       dur     : made duration matrix of 47 frames x 5 features
2017-05-03 17:47:29,372 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0059.lab, 140 labels
2017-05-03 17:47:29,373 [1;34mDEBUG   [0m       dur     : made duration matrix of 28 frames x 5 features
2017-05-03 17:47:29,374 [1;32mINFO    [0m       dur     : loaded /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/label_state_align/arctic_a0060.lab, 110 labels
2017-05-03 17:47:29,375 [1;34mDEBUG   [0m       dur     : made duration matrix of 22 frames x 5 features
2017-05-03 17:47:29,375 [1;32mINFO    [0m       main    : creating acoustic (output) features
2017-05-03 17:47:29,375 [1;32mINFO    [0m  acoustic_comp: processing file    1 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0001.cmp
2017-05-03 17:47:29,375 [1;34mDEBUG   [0m  acoustic_comp:  wrote 37 frames of features
2017-05-03 17:47:29,376 [1;32mINFO    [0m  acoustic_comp: processing file    2 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0002.cmp
2017-05-03 17:47:29,376 [1;34mDEBUG   [0m  acoustic_comp:  wrote 42 frames of features
2017-05-03 17:47:29,376 [1;32mINFO    [0m  acoustic_comp: processing file    3 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0003.cmp
2017-05-03 17:47:29,376 [1;34mDEBUG   [0m  acoustic_comp:  wrote 41 frames of features
2017-05-03 17:47:29,376 [1;32mINFO    [0m  acoustic_comp: processing file    4 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0004.cmp
2017-05-03 17:47:29,376 [1;34mDEBUG   [0m  acoustic_comp:  wrote 28 frames of features
2017-05-03 17:47:29,376 [1;32mINFO    [0m  acoustic_comp: processing file    5 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0005.cmp
2017-05-03 17:47:29,376 [1;34mDEBUG   [0m  acoustic_comp:  wrote 17 frames of features
2017-05-03 17:47:29,376 [1;32mINFO    [0m  acoustic_comp: processing file    6 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0006.cmp
2017-05-03 17:47:29,377 [1;34mDEBUG   [0m  acoustic_comp:  wrote 33 frames of features
2017-05-03 17:47:29,377 [1;32mINFO    [0m  acoustic_comp: processing file    7 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0007.cmp
2017-05-03 17:47:29,377 [1;34mDEBUG   [0m  acoustic_comp:  wrote 40 frames of features
2017-05-03 17:47:29,377 [1;32mINFO    [0m  acoustic_comp: processing file    8 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0008.cmp
2017-05-03 17:47:29,377 [1;34mDEBUG   [0m  acoustic_comp:  wrote 24 frames of features
2017-05-03 17:47:29,377 [1;32mINFO    [0m  acoustic_comp: processing file    9 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0009.cmp
2017-05-03 17:47:29,377 [1;34mDEBUG   [0m  acoustic_comp:  wrote 40 frames of features
2017-05-03 17:47:29,377 [1;32mINFO    [0m  acoustic_comp: processing file   10 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0010.cmp
2017-05-03 17:47:29,377 [1;34mDEBUG   [0m  acoustic_comp:  wrote 41 frames of features
2017-05-03 17:47:29,377 [1;32mINFO    [0m  acoustic_comp: processing file   11 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0011.cmp
2017-05-03 17:47:29,378 [1;34mDEBUG   [0m  acoustic_comp:  wrote 34 frames of features
2017-05-03 17:47:29,378 [1;32mINFO    [0m  acoustic_comp: processing file   12 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0012.cmp
2017-05-03 17:47:29,378 [1;34mDEBUG   [0m  acoustic_comp:  wrote 36 frames of features
2017-05-03 17:47:29,378 [1;32mINFO    [0m  acoustic_comp: processing file   13 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0013.cmp
2017-05-03 17:47:29,378 [1;34mDEBUG   [0m  acoustic_comp:  wrote 52 frames of features
2017-05-03 17:47:29,378 [1;32mINFO    [0m  acoustic_comp: processing file   14 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0014.cmp
2017-05-03 17:47:29,378 [1;34mDEBUG   [0m  acoustic_comp:  wrote 36 frames of features
2017-05-03 17:47:29,378 [1;32mINFO    [0m  acoustic_comp: processing file   15 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0015.cmp
2017-05-03 17:47:29,378 [1;34mDEBUG   [0m  acoustic_comp:  wrote 19 frames of features
2017-05-03 17:47:29,378 [1;32mINFO    [0m  acoustic_comp: processing file   16 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0016.cmp
2017-05-03 17:47:29,379 [1;34mDEBUG   [0m  acoustic_comp:  wrote 41 frames of features
2017-05-03 17:47:29,379 [1;32mINFO    [0m  acoustic_comp: processing file   17 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0017.cmp
2017-05-03 17:47:29,379 [1;34mDEBUG   [0m  acoustic_comp:  wrote 55 frames of features
2017-05-03 17:47:29,379 [1;32mINFO    [0m  acoustic_comp: processing file   18 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0018.cmp
2017-05-03 17:47:29,379 [1;34mDEBUG   [0m  acoustic_comp:  wrote 15 frames of features
2017-05-03 17:47:29,379 [1;32mINFO    [0m  acoustic_comp: processing file   19 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0019.cmp
2017-05-03 17:47:29,379 [1;34mDEBUG   [0m  acoustic_comp:  wrote 44 frames of features
2017-05-03 17:47:29,379 [1;32mINFO    [0m  acoustic_comp: processing file   20 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0020.cmp
2017-05-03 17:47:29,379 [1;34mDEBUG   [0m  acoustic_comp:  wrote 39 frames of features
2017-05-03 17:47:29,379 [1;32mINFO    [0m  acoustic_comp: processing file   21 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0021.cmp
2017-05-03 17:47:29,380 [1;34mDEBUG   [0m  acoustic_comp:  wrote 32 frames of features
2017-05-03 17:47:29,380 [1;32mINFO    [0m  acoustic_comp: processing file   22 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0022.cmp
2017-05-03 17:47:29,380 [1;34mDEBUG   [0m  acoustic_comp:  wrote 54 frames of features
2017-05-03 17:47:29,380 [1;32mINFO    [0m  acoustic_comp: processing file   23 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0023.cmp
2017-05-03 17:47:29,380 [1;34mDEBUG   [0m  acoustic_comp:  wrote 72 frames of features
2017-05-03 17:47:29,380 [1;32mINFO    [0m  acoustic_comp: processing file   24 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0024.cmp
2017-05-03 17:47:29,380 [1;34mDEBUG   [0m  acoustic_comp:  wrote 49 frames of features
2017-05-03 17:47:29,380 [1;32mINFO    [0m  acoustic_comp: processing file   25 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0025.cmp
2017-05-03 17:47:29,380 [1;34mDEBUG   [0m  acoustic_comp:  wrote 38 frames of features
2017-05-03 17:47:29,381 [1;32mINFO    [0m  acoustic_comp: processing file   26 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0026.cmp
2017-05-03 17:47:29,381 [1;34mDEBUG   [0m  acoustic_comp:  wrote 37 frames of features
2017-05-03 17:47:29,381 [1;32mINFO    [0m  acoustic_comp: processing file   27 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0027.cmp
2017-05-03 17:47:29,381 [1;34mDEBUG   [0m  acoustic_comp:  wrote 48 frames of features
2017-05-03 17:47:29,381 [1;32mINFO    [0m  acoustic_comp: processing file   28 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0028.cmp
2017-05-03 17:47:29,381 [1;34mDEBUG   [0m  acoustic_comp:  wrote 17 frames of features
2017-05-03 17:47:29,381 [1;32mINFO    [0m  acoustic_comp: processing file   29 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0029.cmp
2017-05-03 17:47:29,381 [1;34mDEBUG   [0m  acoustic_comp:  wrote 40 frames of features
2017-05-03 17:47:29,381 [1;32mINFO    [0m  acoustic_comp: processing file   30 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0030.cmp
2017-05-03 17:47:29,381 [1;34mDEBUG   [0m  acoustic_comp:  wrote 14 frames of features
2017-05-03 17:47:29,382 [1;32mINFO    [0m  acoustic_comp: processing file   31 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0031.cmp
2017-05-03 17:47:29,382 [1;34mDEBUG   [0m  acoustic_comp:  wrote 26 frames of features
2017-05-03 17:47:29,382 [1;32mINFO    [0m  acoustic_comp: processing file   32 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0032.cmp
2017-05-03 17:47:29,382 [1;34mDEBUG   [0m  acoustic_comp:  wrote 48 frames of features
2017-05-03 17:47:29,382 [1;32mINFO    [0m  acoustic_comp: processing file   33 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0033.cmp
2017-05-03 17:47:29,382 [1;34mDEBUG   [0m  acoustic_comp:  wrote 48 frames of features
2017-05-03 17:47:29,382 [1;32mINFO    [0m  acoustic_comp: processing file   34 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0034.cmp
2017-05-03 17:47:29,382 [1;34mDEBUG   [0m  acoustic_comp:  wrote 44 frames of features
2017-05-03 17:47:29,382 [1;32mINFO    [0m  acoustic_comp: processing file   35 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0035.cmp
2017-05-03 17:47:29,383 [1;34mDEBUG   [0m  acoustic_comp:  wrote 47 frames of features
2017-05-03 17:47:29,383 [1;32mINFO    [0m  acoustic_comp: processing file   36 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0036.cmp
2017-05-03 17:47:29,383 [1;34mDEBUG   [0m  acoustic_comp:  wrote 19 frames of features
2017-05-03 17:47:29,383 [1;32mINFO    [0m  acoustic_comp: processing file   37 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0037.cmp
2017-05-03 17:47:29,383 [1;34mDEBUG   [0m  acoustic_comp:  wrote 28 frames of features
2017-05-03 17:47:29,383 [1;32mINFO    [0m  acoustic_comp: processing file   38 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0038.cmp
2017-05-03 17:47:29,383 [1;34mDEBUG   [0m  acoustic_comp:  wrote 21 frames of features
2017-05-03 17:47:29,383 [1;32mINFO    [0m  acoustic_comp: processing file   39 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0039.cmp
2017-05-03 17:47:29,383 [1;34mDEBUG   [0m  acoustic_comp:  wrote 31 frames of features
2017-05-03 17:47:29,383 [1;32mINFO    [0m  acoustic_comp: processing file   40 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0040.cmp
2017-05-03 17:47:29,384 [1;34mDEBUG   [0m  acoustic_comp:  wrote 31 frames of features
2017-05-03 17:47:29,384 [1;32mINFO    [0m  acoustic_comp: processing file   41 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0041.cmp
2017-05-03 17:47:29,384 [1;34mDEBUG   [0m  acoustic_comp:  wrote 25 frames of features
2017-05-03 17:47:29,384 [1;32mINFO    [0m  acoustic_comp: processing file   42 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0042.cmp
2017-05-03 17:47:29,384 [1;34mDEBUG   [0m  acoustic_comp:  wrote 34 frames of features
2017-05-03 17:47:29,384 [1;32mINFO    [0m  acoustic_comp: processing file   43 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0043.cmp
2017-05-03 17:47:29,384 [1;34mDEBUG   [0m  acoustic_comp:  wrote 46 frames of features
2017-05-03 17:47:29,384 [1;32mINFO    [0m  acoustic_comp: processing file   44 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0044.cmp
2017-05-03 17:47:29,384 [1;34mDEBUG   [0m  acoustic_comp:  wrote 39 frames of features
2017-05-03 17:47:29,384 [1;32mINFO    [0m  acoustic_comp: processing file   45 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0045.cmp
2017-05-03 17:47:29,385 [1;34mDEBUG   [0m  acoustic_comp:  wrote 30 frames of features
2017-05-03 17:47:29,385 [1;32mINFO    [0m  acoustic_comp: processing file   46 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0046.cmp
2017-05-03 17:47:29,385 [1;34mDEBUG   [0m  acoustic_comp:  wrote 35 frames of features
2017-05-03 17:47:29,385 [1;32mINFO    [0m  acoustic_comp: processing file   47 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0047.cmp
2017-05-03 17:47:29,385 [1;34mDEBUG   [0m  acoustic_comp:  wrote 39 frames of features
2017-05-03 17:47:29,385 [1;32mINFO    [0m  acoustic_comp: processing file   48 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0048.cmp
2017-05-03 17:47:29,385 [1;34mDEBUG   [0m  acoustic_comp:  wrote 30 frames of features
2017-05-03 17:47:29,385 [1;32mINFO    [0m  acoustic_comp: processing file   49 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0049.cmp
2017-05-03 17:47:29,385 [1;34mDEBUG   [0m  acoustic_comp:  wrote 36 frames of features
2017-05-03 17:47:29,385 [1;32mINFO    [0m  acoustic_comp: processing file   50 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0050.cmp
2017-05-03 17:47:29,386 [1;34mDEBUG   [0m  acoustic_comp:  wrote 48 frames of features
2017-05-03 17:47:29,386 [1;32mINFO    [0m  acoustic_comp: processing file   51 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0051.cmp
2017-05-03 17:47:29,386 [1;34mDEBUG   [0m  acoustic_comp:  wrote 47 frames of features
2017-05-03 17:47:29,386 [1;32mINFO    [0m  acoustic_comp: processing file   52 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0052.cmp
2017-05-03 17:47:29,386 [1;34mDEBUG   [0m  acoustic_comp:  wrote 25 frames of features
2017-05-03 17:47:29,386 [1;32mINFO    [0m  acoustic_comp: processing file   53 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0053.cmp
2017-05-03 17:47:29,386 [1;34mDEBUG   [0m  acoustic_comp:  wrote 41 frames of features
2017-05-03 17:47:29,386 [1;32mINFO    [0m  acoustic_comp: processing file   54 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0054.cmp
2017-05-03 17:47:29,386 [1;34mDEBUG   [0m  acoustic_comp:  wrote 20 frames of features
2017-05-03 17:47:29,386 [1;32mINFO    [0m  acoustic_comp: processing file   55 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0055.cmp
2017-05-03 17:47:29,387 [1;34mDEBUG   [0m  acoustic_comp:  wrote 50 frames of features
2017-05-03 17:47:29,387 [1;32mINFO    [0m  acoustic_comp: processing file   56 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0056.cmp
2017-05-03 17:47:29,387 [1;34mDEBUG   [0m  acoustic_comp:  wrote 33 frames of features
2017-05-03 17:47:29,387 [1;32mINFO    [0m  acoustic_comp: processing file   57 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0057.cmp
2017-05-03 17:47:29,387 [1;34mDEBUG   [0m  acoustic_comp:  wrote 23 frames of features
2017-05-03 17:47:29,387 [1;32mINFO    [0m  acoustic_comp: processing file   58 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0058.cmp
2017-05-03 17:47:29,387 [1;34mDEBUG   [0m  acoustic_comp:  wrote 47 frames of features
2017-05-03 17:47:29,387 [1;32mINFO    [0m  acoustic_comp: processing file   59 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0059.cmp
2017-05-03 17:47:29,387 [1;34mDEBUG   [0m  acoustic_comp:  wrote 28 frames of features
2017-05-03 17:47:29,388 [1;32mINFO    [0m  acoustic_comp: processing file   60 of   60 : /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/nn_dur_5/arctic_a0060.cmp
2017-05-03 17:47:29,388 [1;34mDEBUG   [0m  acoustic_comp:  wrote 22 frames of features
2017-05-03 17:47:29,465 [1;32mINFO    [0m       main    : normalising acoustic (output) features using method MVN
/data/joniva/software/merlin/src/frontend/mean_variance_norm.py:69: FutureWarning: comparison to `None` will result in an elementwise object comparison in the future.
  if self.mean_vector == None:
/data/joniva/software/merlin/src/frontend/mean_variance_norm.py:71: FutureWarning: comparison to `None` will result in an elementwise object comparison in the future.
  if self.std_vector  == None:
2017-05-03 17:47:29,475 [1;32mINFO    [0m       main    : saved MVN vectors to /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/norm_info_dur_5_MVN.dat
2017-05-03 17:47:29,476 [1;32mINFO    [0m       main    : saved dur variance vector to /data/joniva/software/merlin/egs/slt_arctic/s1/experiments/slt_arctic_demo/duration_model/data/var/dur_5
2017-05-03 17:47:29,529 [1;34mDEBUG   [0m       labels  : HTS-derived input feature dimension is 1483 + 0 = 1483
2017-05-03 17:47:29,529 [1;32mINFO    [0m       main    : label dimension is 1483
2017-05-03 17:47:29,529 [1;32mINFO    [0m       main    : training DNN
2017-05-03 17:47:29,530 [1;34mDEBUG   [0m main.train_DNN: Starting train_DNN
2017-05-03 17:47:29,530 [1;34mDEBUG   [0m main.train_DNN: Creating training   data provider
2017-05-03 17:47:29,530 [1;34mDEBUG   [0m main.train_DNN: Creating validation data provider
2017-05-03 17:47:29,553 [1;32mINFO    [0m main.train_DNN: building the model
2017-05-03 17:47:30,422 [1;32mINFO    [0m main.train_DNN: fine-tuning the DNN model
0.002
2017-05-03 17:47:30,485 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:30,488 [1;32mINFO    [0m main.train_DNN: epoch 1, validation error 4.757449, train error 4.923875  time spent 0.07
2017-05-03 17:47:30,549 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:30,552 [1;32mINFO    [0m main.train_DNN: epoch 2, validation error 4.617240, train error 4.623313  time spent 0.06
2017-05-03 17:47:30,612 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:30,615 [1;32mINFO    [0m main.train_DNN: epoch 3, validation error 4.526011, train error 4.448405  time spent 0.06
2017-05-03 17:47:30,675 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:30,678 [1;32mINFO    [0m main.train_DNN: epoch 4, validation error 4.454467, train error 4.321902  time spent 0.06
2017-05-03 17:47:30,738 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:30,741 [1;32mINFO    [0m main.train_DNN: epoch 5, validation error 4.395353, train error 4.220609  time spent 0.06
2017-05-03 17:47:30,801 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:30,804 [1;32mINFO    [0m main.train_DNN: epoch 6, validation error 4.346066, train error 4.136137  time spent 0.06
2017-05-03 17:47:32,689 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:32,692 [1;32mINFO    [0m main.train_DNN: epoch 7, validation error 4.305002, train error 4.064638  time spent 0.08
2017-05-03 17:47:34,553 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:34,556 [1;32mINFO    [0m main.train_DNN: epoch 8, validation error 4.271884, train error 4.003446  time spent 0.07
2017-05-03 17:47:36,423 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:36,427 [1;32mINFO    [0m main.train_DNN: epoch 9, validation error 4.247316, train error 3.950342  time spent 0.07
2017-05-03 17:47:38,316 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:38,320 [1;32mINFO    [0m main.train_DNN: epoch 10, validation error 4.231430, train error 3.903510  time spent 0.07
2017-05-03 17:47:40,248 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:40,251 [1;32mINFO    [0m main.train_DNN: epoch 11, validation error 4.184093, train error 3.895158  time spent 0.14
2017-05-03 17:47:42,118 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:42,122 [1;32mINFO    [0m main.train_DNN: epoch 12, validation error 4.187752, train error 3.796293  time spent 0.07
2017-05-03 17:47:42,122 [1;34mDEBUG   [0m main.train_DNN: validation loss increased
2017-05-03 17:47:42,185 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:42,188 [1;32mINFO    [0m main.train_DNN: epoch 13, validation error 4.196585, train error 3.673666  time spent 0.07
2017-05-03 17:47:42,188 [1;34mDEBUG   [0m main.train_DNN: validation loss increased
2017-05-03 17:47:42,248 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:42,251 [1;32mINFO    [0m main.train_DNN: epoch 14, validation error 4.192706, train error 3.591846  time spent 0.06
2017-05-03 17:47:42,311 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:42,314 [1;32mINFO    [0m main.train_DNN: epoch 15, validation error 4.195555, train error 3.560694  time spent 0.06
2017-05-03 17:47:42,314 [1;34mDEBUG   [0m main.train_DNN: validation loss increased
2017-05-03 17:47:42,376 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:42,379 [1;32mINFO    [0m main.train_DNN: epoch 16, validation error 4.208522, train error 3.529484  time spent 0.06
2017-05-03 17:47:42,379 [1;34mDEBUG   [0m main.train_DNN: validation loss increased
2017-05-03 17:47:42,509 [1;34mDEBUG   [0m main.train_DNN: calculating validation loss
2017-05-03 17:47:42,512 [1;32mINFO    [0m main.train_DNN: epoch 17, validation error 4.181202, train error 3.510999  time spent 0.13
2017-05-03 17:47:42,840 [1;35mCRITICAL[0m       main    : train_DNN interrupted via keyboard
Lock freed
 
Failed tests:  charac_slt_short_merlin_execution
