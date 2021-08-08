class ArgumentManager:
    """
    For OLS's arguments input management
    """
    app_id = 1692080
    pro_dlc_id = 1718750

    """Release Version Control"""
    is_steam = True
    is_free = True
    version_tag = "3.5.1 alpha"
    """ 发布前改动以上参数即可 """

    def __init__(self, args: dict):
        self.app_dir = args.get("app_dir", "")
        self.ols_path = args.get("ols_path", "")
        self.batch = args.get("batch", False)
        self.ffmpeg = args.get("ffmpeg", "")

        self.config = args.get("config", "")
        self.input = args.get("input", "")
        self.output_dir = args.get("output_dir", "")
        self.task_id = args.get("task_id", "")
        self.gui_inputs = args.get("gui_inputs", "")
        self.input_fps = args.get("input_fps", 0)
        self.target_fps = args.get("target_fps", 0)
        self.output_ext = args.get("output_ext", ".mp4")
        self.is_img_input = args.get("is_img_input", False)
        self.is_img_output = args.get("is_img_output", False)
        self.is_output_only = args.get("is_output_only", True)
        self.is_save_audio = args.get("is_save_audio", True)
        self.input_start_point = args.get("input_start_point", None)
        self.input_end_point = args.get("input_end_point", None)
        self.output_chunk_cnt = args.get("output_chunk_cnt", 0)
        self.interp_start = args.get("interp_start", 0)

        self.is_no_scdet = args.get("is_no_scdet", False)
        self.is_scdet_mix = args.get("is_scdet_mix", False)
        self.use_scdet_fixed = args.get("use_scdet_fixed", False)
        self.is_scdet_output = args.get("is_scdet_output", True)
        self.scdet_threshold = args.get("scdet_threshold", 10)
        self.scdet_fixed_max = args.get("scdet_fixed_max", 40)
        self.scdet_flow_cnt = args.get("scdet_flow_cnt", 4)
        self.scdet_mode = args.get("scdet_mode", 0)
        self.remove_dup_mode = args.get("remove_dup_mode", 0)
        self.remove_dup_threshold = args.get("remove_dup_threshold", 0.1)

        self.use_manual_buffer = args.get("use_manual_buffer", False)
        self.manual_buffer_size = args.get("manual_buffer_size", 1)

        self.resize_width = args.get("resize_width", "")
        self.resize_height = args.get("resize_height", "")
        self.resize = args.get("resize", "")
        self.resize_exp = args.get("resize_exp", 1)
        self.crop_width = args.get("crop_width", "")
        self.crop_height = args.get("crop_height", "")
        self.crop = args.get("crop", "")

        self.use_sr = args.get("use_sr", False)
        self.use_sr_algo = args.get("use_sr_algo", "")
        self.use_sr_model = args.get("use_sr_model", "")
        self.use_sr_mode = args.get("use_sr_mode", "")
        self.sr_tilesize = args.get("sr_tilesize", 200)

        self.render_gap = args.get("render_gap", 1000)
        self.use_crf = args.get("use_crf", True)
        self.use_bitrate = args.get("use_bitrate", False)
        self.render_crf = args.get("render_crf", 14)
        self.render_bitrate = args.get("render_bitrate", 90)
        self.render_encoder_preset = args.get("render_encoder_preset", "slow")
        self.render_encoder = args.get("render_encoder", "")
        self.render_hwaccel_mode = args.get("render_hwaccel_mode", "")
        self.render_hwaccel_preset = args.get("render_hwaccel_preset", "")
        self.use_hwaccel_decode = args.get("use_hwaccel_decode", True)
        self.use_manual_encode_thread = args.get("use_manual_encode_thread", False)
        self.render_encode_thread = args.get("render_encode_thread", 16)
        self.is_quick_extract = args.get("is_quick_extract", True)
        self.is_hdr_strict_mode = args.get("is_hdr_strict_mode", False)
        self.render_ffmpeg_customized = args.get("render_ffmpeg_customized", "")
        self.is_no_concat = args.get("is_no_concat", False)
        self.use_fast_denoise = args.get("use_fast_denoise", False)
        self.gif_loop = args.get("gif_loop", True)
        self.is_render_slow_motion = args.get("is_render_slow_motion", False)
        self.render_slow_motion_fps = args.get("render_slow_motion_fps", 0)
        self.use_deinterlace = args.get("use_deinterlace", False)

        self.use_ncnn = args.get("use_ncnn", False)
        self.ncnn_thread = args.get("ncnn_thread", 4)
        self.ncnn_gpu = args.get("ncnn_gpu", 0)
        self.use_rife_tta_mode = args.get("use_rife_tta_mode", False)
        self.use_rife_fp16 = args.get("use_rife_fp16", False)
        self.rife_scale = args.get("rife_scale", 1.0)
        self.rife_model_dir = args.get("rife_model_dir", "")
        self.rife_model = args.get("rife_model", "")
        self.rife_model_name = args.get("rife_model_name", "")
        self.rife_exp = args.get("rife_exp", 1.0)
        self.rife_cuda_cnt = args.get("rife_cuda_cnt", 0)
        self.is_rife_reverse = args.get("is_rife_reverse", False)
        self.use_specific_gpu = args.get("use_specific_gpu", 0)  # !
        self.use_rife_auto_scale = args.get("use_rife_auto_scale", False)
        self.use_rife_forward_ensemble = args.get("use_rife_forward_ensemble", False)
        self.use_rife_multi_cards = args.get("use_rife_multi_cards", False)

        self.debug = args.get("debug", False)
        self.multi_task_rest = args.get("multi_task_rest", False)
        self.multi_task_rest_interval = args.get("multi_task_rest_interval", 1)
        self.after_mission = args.get("after_mission", False)
        self.force_cpu = args.get("force_cpu", False)
        self.expert_mode = args.get("expert_mode", False)
        self.preview_args = args.get("preview_args", False)
        self.pos = args.get("pos", "")
        self.size = args.get("size", "")

        """OLS Params"""
        self.concat_only = args.get("concat_only", False)
        self.extract_only = args.get("extract_only", False)
        self.render_only = args.get("render_only", False)
        self.version = args.get("version", "0.0.0 beta")
