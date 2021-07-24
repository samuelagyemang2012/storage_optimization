#!/bin/bash
declare -a params

echo "Generating params"
mds_cache_memory_limit=$[ $RANDOM % 1073741824 + 5073741824]
params[0]=$mds_cache_memory_limit

mds_cache_reservation=$(awk -v min=0.05 -v max=1.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[1]=$mds_cache_reservation

mds_health_cache_threshold=$(awk -v min=1.5 -v max=5.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[2]=$mds_health_cache_threshold

mds_cache_trim_threshold=$[ $RANDOM % 65536 + 85000]
params[3]=$mds_cache_trim_threshold

mds_cache_trim_decay_rate=$(awk -v min=1.0 -v max=5.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[4]=$mds_cache_trim_decay_rate

mds_recall_max_caps=$[ $RANDOM % 5000 + 8000]
params[5]=$mds_recall_max_caps

mds_recall_max_decay_threshold=$[ $RANDOM % 16384 + 28000]
params[6]=$mds_recall_max_decay_threshold

mds_recall_max_decay_rate=$(awk -v min=2.5 -v max=7.5 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[7]=$mds_recall_max_decay_rate

mds_recall_global_max_decay_threshold=$[ $RANDOM % 65536 + 85000]
params[8]=$mds_recall_global_max_decay_threshold

mds_recall_warning_threshold=$[ $RANDOM % 32768 + 86000]
params[9]=$mds_recall_warning_threshold

mds_recall_warning_decay_rate=$(awk -v min=60.0 -v max=80.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[10]=$mds_recall_warning_decay_rate

mds_session_cap_acquisition_throttle=$[ $RANDOM % 500000 + 1000000]
params[11]=$mds_session_cap_acquisition_throttle

mds_session_cap_acquisition_decay_rate=$(awk -v min=10.0 -v max=20.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[12]=$mds_session_cap_acquisition_decay_rate

mds_session_max_caps_throttle_ratio=$(awk -v min=1.1 -v max=4.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[13]=$mds_session_max_caps_throttle_ratio

mds_cap_acquisition_throttle_retry_request_timeout=$(awk -v min=0.5 -v max=4.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[14]=$mds_cap_acquisition_throttle_retry_request_timeout

mds_session_cache_liveness_magnitude=$[ $RANDOM % 10 + 20]
params[15]=$mds_session_cache_liveness_magnitude

mds_session_cache_liveness_decay_rate=$(awk -v min=5.0 -v max=10.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[16]=$mds_session_cache_liveness_decay_rate

mds_max_caps_per_client=$[ $RANDOM % 1048576 + 3000000]
params[17]=$mds_max_caps_per_client

mds_cache_mid=$(awk -v min=0.7 -v max=4.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[18]=$mds_cache_mid

mds_dir_max_commit_size=$[ $RANDOM % 10 + 30]
params[19]=$mds_dir_max_commit_size

mds_decay_halflife=$(awk -v min=5.0 -v max=25.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[20]=$mds_decay_halflife

mds_beacon_interval=$(awk -v min=4.0 -v max=20.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[21]=$mds_beacon_interval

mds_beacon_grace=$(awk -v min=15.0 -v max=50.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[22]=$mds_beacon_grace

mon_mds_blocklist_interval=$(awk -v min=3600.0 -v max=86400.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[23]=$mon_mds_blocklist_interval

mds_reconnect_timeout=$(awk -v min=45.0 -v max=120.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[24]=$mds_reconnect_timeout

mds_tick_interval=$(awk -v min=5.0 -v max=25.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[25]=$mds_tick_interval

mds_dirstat_min_interval=$(awk -v min=1.0 -v max=15.5 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[26]=$mds_dirstat_min_interval

mds_scatter_nudge_interval=$(awk -v min=5.0 -v max=20.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[27]=$mds_scatter_nudge_interval

mds_client_prealloc_inos=$[ $RANDOM % 1000 + 3000]
params[28]=$mds_client_prealloc_inos

mds_early_reply=$[ $RANDOM % 2]
params[29]=$mds_early_reply

mds_default_dir_hash=$[ $RANDOM % 2 + 10]
params[30]=$mds_default_dir_hash

mds_log_skip_corrupt_events=$[ $RANDOM % 2]
params[31]=$mds_log_skip_corrupt_events

mds_log_max_events=$[ $RANDOM % -1 + 5]
params[32]=$mds_log_max_events

mds_log_max_segments=$[ $RANDOM % 128 + 2048]
params[33]=$mds_log_max_segments

mds_bal_sample_interval=$(awk -v min=3.0 -v max=10.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[34]=$mds_bal_sample_interval

mds_bal_replicate_threshold=$(awk -v min=8000.0 -v max=15000.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[35]=$mds_bal_replicate_threshold

mds_bal_unreplicate_threshold=$(awk -v min=0.0 -v max=5.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[36]=$mds_bal_unreplicate_threshold

mds_bal_split_size=$[ $RANDOM % 10000 + 80000]
params[37]=$mds_bal_split_size

mds_bal_split_rd=$(awk -v min=25000.0 -v max=100000.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[38]=$mds_bal_split_rd

mds_bal_split_wr=$(awk -v min=10000.0 -v max=80000.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[39]=$mds_bal_split_wr

mds_bal_split_bits=$[ $RANDOM % 1 + 24]
params[40]=$mds_bal_split_bits

mds_bal_merge_size=$[ $RANDOM % 50 + 100]
params[41]=$mds_bal_merge_size

mds_bal_interval=$[ $RANDOM % 10 + 50]
params[42]=$mds_bal_interval

mds_bal_fragment_interval=$[ $RANDOM % 5 + 20]
params[43]=$mds_bal_fragment_interval

mds_bal_fragment_fast_factor=$(awk -v min=1.5 -v max=20.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[44]=$mds_bal_fragment_fast_factor

mds_bal_fragment_size_max=$[ $RANDOM % 100000 + 500000]
params[45]=$mds_bal_fragment_size_max

mds_bal_idle_threshold=$(awk -v min=0.0 -v max=12.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[46]=$mds_bal_idle_threshold

mds_bal_max=$[ $RANDOM % -1 + 2]
params[47]=$mds_bal_max

mds_bal_max_until=$[ $RANDOM % -1 + 2]
params[48]=$mds_bal_max_until

mds_bal_mode=$[ $RANDOM % 1 + 3]
params[49]=$mds_bal_mode

mds_bal_min_rebalance=$(awk -v min=0.1 -v max=4.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[50]=$mds_bal_min_rebalance

mds_bal_min_start=$(awk -v min=0.2 -v max=5.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[51]=$mds_bal_min_start

mds_bal_need_min=$(awk -v min=0.8 -v max=5.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[52]=$mds_bal_need_min

mds_bal_need_max=$(awk -v min=1.2 -v max=10.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[53]=$mds_bal_need_max

mds_bal_midchunk=$(awk -v min=0.3 -v max=5.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[54]=$mds_bal_midchunk

mds_bal_minchunk=$(awk -v min=0.001 -v max=1.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[55]=$mds_bal_minchunk

mds_replay_interval=$(awk -v min=1.0 -v max=7.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[56]=$mds_replay_interval

mds_shutdown_check=$[ $RANDOM % 1 + 5]
params[57]=$mds_shutdown_check

mds_thrash_exports=$[ $RANDOM % 1 + 5]
params[58]=$mds_thrash_exports

mds_thrash_fragments=$[ $RANDOM % 1 + 5]
params[59]=$mds_thrash_fragments

mds_dump_cache_on_map=$[ $RANDOM % 2]
params[60]=$mds_dump_cache_on_map

mds_dump_cache_after_rejoin=$[ $RANDOM % 2]
params[61]=$mds_dump_cache_after_rejoin

mds_verify_scatter=$[ $RANDOM % 2]
params[62]=$mds_verify_scatter

mds_debug_scatterstat=$[ $RANDOM % 2]
params[63]=$mds_debug_scatterstat

mds_debug_frag=$[ $RANDOM % 2]
params[64]=$mds_debug_frag

mds_debug_auth_pins=$[ $RANDOM % 2]
params[65]=$mds_debug_auth_pins

mds_debug_subtrees=$[ $RANDOM % 2]
params[66]=$mds_debug_subtrees

mds_kill_mdstable_at=$[ $RANDOM % 1 + 6]
params[67]=$mds_kill_mdstable_at

mds_kill_export_at=$[ $RANDOM % 1 + 6]
params[68]=$mds_kill_export_at

mds_kill_import_at=$[ $RANDOM % 1 + 6]
params[69]=$mds_kill_import_at

mds_kill_link_at=$[ $RANDOM % 1 + 5]
params[70]=$mds_kill_link_at

mds_kill_rename_at=$[ $RANDOM % 1 + 5]
params[71]=$mds_kill_rename_at

mds_wipe_sessions=$[ $RANDOM % 2]
params[72]=$mds_wipe_sessions

mds_wipe_ino_prealloc=$[ $RANDOM % 2]
params[73]=$mds_wipe_ino_prealloc

mds_skip_ino=$[ $RANDOM % 1 + 4]
params[74]=$mds_skip_ino

mds_min_caps_per_client=$[ $RANDOM % 100 + 500]
params[75]=$mds_min_caps_per_client

bluestore_cache_trim_interval=$(awk -v min=0.05 -v max=1.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[76]=$bluestore_cache_trim_interval

bluestore_default_buffered_read=$[ $RANDOM % 2]
params[77]=$bluestore_default_buffered_read

bluestore_default_buffered_write=$[ $RANDOM % 2]
params[78]=$bluestore_default_buffered_write

bluestore_deferred_batch_ops=$[ $RANDOM % 1 + 64]
params[79]=$bluestore_deferred_batch_ops

bluestore_deferred_batch_ops_hdd=$[ $RANDOM % 64 + 128]
params[80]=$bluestore_deferred_batch_ops_hdd

bluestore_deferred_batch_ops_ssd=$[ $RANDOM % 16 + 128]
params[81]=$bluestore_deferred_batch_ops_ssd

bluestore_fsck_read_bytes_cap=$[ $RANDOM % 67108864 + 87108864]
params[82]=$bluestore_fsck_read_bytes_cap

bluestore_max_deferred_txc=$[ $RANDOM % 32 + 120]
params[83]=$bluestore_max_deferred_txc

bluestore_prefer_deferred_size=$[ $RANDOM % 1 + 24]
params[84]=$bluestore_prefer_deferred_size

bluestore_prefer_deferred_size_hdd=$[ $RANDOM % 32768 + 50000]
params[85]=$bluestore_prefer_deferred_size_hdd

bluestore_prefer_deferred_size_ssd=$[ $RANDOM % 1 + 2400]
params[86]=$bluestore_prefer_deferred_size_ssd

bluestore_throttle_bytes=$[ $RANDOM % 67108864 + 97108864]
params[87]=$bluestore_throttle_bytes

client_cache_mid=$(awk -v min=0.75 -v max=2.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[88]=$client_cache_mid

client_cache_size=$[ $RANDOM % 16384 + 34000]
params[89]=$client_cache_size

client_oc=$[ $RANDOM % 2]
params[90]=$client_oc

client_oc_max_dirty=$[ $RANDOM % 104857600 + 104857600]
params[91]=$client_oc_max_dirty

client_oc_max_dirty_age=$(awk -v min=5.0 -v max=25.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[92]=$client_oc_max_dirty_age

client_oc_max_objects=$[ $RANDOM % 1000 + 10000]
params[93]=$client_oc_max_objects

client_oc_size=$[ $RANDOM % 209715200 + 509715200]
params[94]=$client_oc_size

client_oc_target_dirty=$[ $RANDOM % 8388608 + 12000000]
params[95]=$client_oc_target_dirty

client_readahead_max_bytes=$[ $RANDOM % 1 + 1024]
params[96]=$client_readahead_max_bytes

client_readahead_max_periods=$[ $RANDOM % 4 + 20]
params[97]=$client_readahead_max_periods

client_readahead_min=$[ $RANDOM % 131072 + 450000]
params[98]=$client_readahead_min

ms_async_max_op_threads=$[ $RANDOM % 5 + 24]
params[99]=$ms_async_max_op_threads

ms_async_op_threads=$[ $RANDOM % 1 + 24]
params[100]=$ms_async_op_threads

osd_bench_max_block_size=$[ $RANDOM % 67108864 + 80000000]
params[101]=$osd_bench_max_block_size

osd_bench_small_size_max_iops=$[ $RANDOM % 100 + 500]
params[102]=$osd_bench_small_size_max_iops

osd_pool_default_cache_max_evict_check_size=$[ $RANDOM % 10 + 50]
params[103]=$osd_pool_default_cache_max_evict_check_size

osd_pool_default_cache_min_evict_age=$[ $RANDOM % 1 + 10]
params[104]=$osd_pool_default_cache_min_evict_age

osd_pool_default_cache_min_flush_age=$[ $RANDOM % 1 + 10]
params[105]=$osd_pool_default_cache_min_flush_age

osd_pool_default_cache_target_dirty_high_ratio=$(awk -v min=0.6 -v max=5.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[106]=$osd_pool_default_cache_target_dirty_high_ratio

osd_pool_default_cache_target_dirty_ratio=$(awk -v min=0.4 -v max=4.5 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[107]=$osd_pool_default_cache_target_dirty_ratio

osd_pool_default_cache_target_full_ratio=$(awk -v min=0.8 -v max=4.5 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[108]=$osd_pool_default_cache_target_full_ratio

osd_target_transaction_size=$[ $RANDOM % 30 + 50]
params[109]=$osd_target_transaction_size

osd_tier_default_cache_hit_set_count=$[ $RANDOM % 4 + 8]
params[110]=$osd_tier_default_cache_hit_set_count

osd_tier_default_cache_hit_set_grade_decay_rate=$[ $RANDOM % 20 + 50]
params[111]=$osd_tier_default_cache_hit_set_grade_decay_rate

osd_tier_default_cache_hit_set_period=$[ $RANDOM % 1200 + 2400]
params[112]=$osd_tier_default_cache_hit_set_period

osd_tier_default_cache_hit_set_search_last_n=$[ $RANDOM % 1 + 5]
params[113]=$osd_tier_default_cache_hit_set_search_last_n

osd_tier_default_cache_min_read_recency_for_promote=$[ $RANDOM % 1 + 5]
params[114]=$osd_tier_default_cache_min_read_recency_for_promote

osd_tier_default_cache_min_write_recency_for_promote=$[ $RANDOM % 1 + 5]
params[115]=$osd_tier_default_cache_min_write_recency_for_promote

rbd_cache=$[ $RANDOM % 2]
params[116]=$rbd_cache

rbd_cache_block_writes_upfront=$[ $RANDOM % 2]
params[117]=$rbd_cache_block_writes_upfront

rbd_cache_max_dirty=$[ $RANDOM % 25165824 + 50000000]
params[118]=$rbd_cache_max_dirty

rbd_cache_max_dirty_age=$(awk -v min=1.0 -v max=5.0 'BEGIN{srand(); print min+rand()*(max-min+1)}')
params[119]=$rbd_cache_max_dirty_age

rbd_cache_max_dirty_object=$[ $RANDOM % 1 + 5]
params[120]=$rbd_cache_max_dirty_object

rbd_cache_size=$[ $RANDOM % 33554432 + 5120000]
params[121]=$rbd_cache_size

rbd_cache_target_dirty=$[ $RANDOM % 16777216 + 24000000]
params[122]=$rbd_cache_target_dirty

rbd_cache_writethrough_until_flush=$[ $RANDOM % 2]
params[123]=$rbd_cache_writethrough_until_flush

rbd_enable_alloc_hint=$[ $RANDOM % 2]
params[124]=$rbd_enable_alloc_hint

rbd_non_blocking_aio=$[ $RANDOM % 2]
params[125]=$rbd_non_blocking_aio

rbd_readahead_disable_after_bytes=$[ $RANDOM % 52428800 + 80000000]
params[126]=$rbd_readahead_disable_after_bytes

rbd_readahead_max_bytes=$[ $RANDOM % 524288 + 1240000]
params[127]=$rbd_readahead_max_bytes

echo "Generating ceph config"
echo "">/etc/ceph/ceph.conf
echo "# minimal ceph.conf for 8fbf4e36-e6c6-11eb-91ce-6bc35296a5da">>/etc/ceph/ceph.conf
echo "[global]">>/etc/ceph/ceph.conf
echo "fsid = 8fbf4e36-e6c6-11eb-91ce-6bc35296a5da">>/etc/ceph/ceph.conf
echo "mon_host = [v2:172.19.203.10:3300/0,v1:172.19.203.10:6789/0]">>/etc/ceph/ceph.conf

n=0
while read line;do 
echo $line = ${params[n]}>>/etc/ceph/ceph.conf
echo $line = ${params[n]}>>gen.txt
#echo $line
n=$((n+1))
done <params.txt
echo "Rados bench-marking with params"
rados bench -p paraTune 10 seq >> rados_results.txt
echo "Done"

