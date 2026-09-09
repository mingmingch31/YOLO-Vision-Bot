use screenshots::Screen;
use std::time::Duration;
use std::thread;
use enigo::{Enigo, MouseControllable, MouseButton};
//use rand::Rng;
use device_query::{DeviceQuery, DeviceState, Keycode};
use crossbeam::channel::unbounded;

use ort::session::Session;
use ort::value::Value;
//use ort::ExecutionProviderDispatch;
use image::{ImageBuffer, Rgba, imageops, imageops::FilterType};

fn main() {
    println!("【智能类人视觉大脑】GPU 异构加速版已加载！");
    println!("按下 [F1] 键 -> 随时 启动/暂停");
    println!("长按 [ESC] 键 -> 彻底关闭程序");

    let _ = ort::init().with_name("YOLO_Vision").commit();

    // ==========================================
    // 核心改造 1：向引擎注入 CUDA 硬件加速指令
    // ==========================================
    let mut model = Session::builder().unwrap()
        .with_intra_threads(4).unwrap()
        // 使用新版架构挂载 CUDA 显卡算力
        // 稳定版的标准 CUDA 挂载语法
        .with_execution_providers([ort::ep::CUDA::default().build()]).unwrap()
        .commit_from_file("models/yolov8n.onnx")
        .expect("未能找到 yolov8n.onnx");
        
    println!("✅ AI 视觉大脑唤醒成功！显卡通道已开启！");
    println!("--------------------------------------------------");

    let (sender, receiver) = unbounded();

    thread::spawn(move || {
        let device_state = DeviceState::new();
        let mut f1_was_pressed = false;
        loop {
            let keys = device_state.get_keys();
            if keys.contains(&Keycode::Escape) {
                let _ = sender.send("ESC");
                break; 
            }
            let f1_is_pressed = keys.contains(&Keycode::F1);
            if f1_is_pressed && !f1_was_pressed {
                let _ = sender.send("TOGGLE"); 
            }
            f1_was_pressed = f1_is_pressed;
            thread::sleep(Duration::from_millis(10));
        }
    });

    let screens = Screen::all().unwrap();
    let primary_screen = screens[0];
    let mut enigo = Enigo::new();
    let mut is_active = false;

    loop {
        if let Ok(msg) = receiver.try_recv() {
            if msg == "ESC" { break; } 
            else if msg == "TOGGLE" { is_active = !is_active; }
        }

        if is_active {
            // ==========================================
            // 核心改造 2：物理级高倍狙击镜架构 (精准 ROI + 画质放大)
            // ==========================================
            let screen_w = primary_screen.display_info.width;
            let screen_h = primary_screen.display_info.height;
            
            // 将检测区域极限缩小到准星周围的 320x320
            let roi_size = 320; 
            let roi_x = (screen_w / 2) - (roi_size / 2);
            let roi_y = (screen_h / 2) - (roi_size / 2);

            let image = primary_screen.capture_area(roi_x as i32, roi_y as i32, roi_size as u32, roi_size as u32).unwrap();
            let raw_bytes = image.as_raw();
            let img_buffer: ImageBuffer<Rgba<u8>, _> = ImageBuffer::from_raw(
                roi_size as u32, roi_size as u32, raw_bytes.to_vec()
            ).unwrap();
            
            // 强行用插值算法拉伸回引擎要求的 640x640，等同于开启 2 倍镜，完美放大敌人头部特征
            let resized_img = imageops::resize(&img_buffer, 640, 640, FilterType::Triangle);

            let mut tensor_data = Vec::with_capacity(3 * 640 * 640);
            for p in resized_img.pixels() { tensor_data.push(p[0] as f32 / 255.0); } 
            for p in resized_img.pixels() { tensor_data.push(p[1] as f32 / 255.0); } 
            for p in resized_img.pixels() { tensor_data.push(p[2] as f32 / 255.0); } 
            
            let shape = vec![1, 3, 640, 640];
            let input_tensor = Value::from_array((shape, tensor_data)).unwrap();

            let _outputs = model.run(ort::inputs![input_tensor]).unwrap();
            let (_shape, data) = _outputs[0].try_extract_tensor::<f32>().unwrap();

            let mut best_conf = 0.0_f32;
            let mut target_cx = 0.0_f32;
            let mut target_cy = 0.0_f32;

            for i in 0..8400 {
                let person_conf = data[4 * 8400 + i]; 
                if person_conf > best_conf && person_conf > 0.4 {
                    best_conf = person_conf;
                    target_cx = data[0 * 8400 + i]; 
                    target_cy = data[1 * 8400 + i];
                }
            }

            // ==========================================
            // 核心改造 3：降维映射与极速连发
            // ==========================================
            if best_conf > 0.61 { 
                let human_x = target_cx;
                let human_y = target_cy + 10.0; // 锁定胸腔偏上

                // AI 在 640x640 画布里看到的相对距离
                let ai_delta_x = human_x - 320.0;
                let ai_delta_y = human_y - 320.0;

                // 按照 320 到 640 的两倍缩放比例，还原真实物理鼠标的移动像素
                let scale_factor = roi_size as f32 / 640.0;
                let real_delta_x = ai_delta_x * scale_factor;
                let real_delta_y = ai_delta_y * scale_factor;

                let steps = 2; // 暴力瞬锁，彻底修复停顿
                let step_x = real_delta_x / steps as f32;
                let step_y = real_delta_y / steps as f32;
                
                for _ in 1..=steps {
                    enigo.mouse_move_relative(step_x as i32, step_y as i32);
                }

                // 零延迟开火
                enigo.mouse_click(MouseButton::Left);
                thread::sleep(Duration::from_millis(15)); 
            }
            
            // 极限压榨，仅给 CPU 喘息 1 毫秒
            thread::sleep(Duration::from_millis(1));
        } else {
            thread::sleep(Duration::from_millis(100));
        }
    }
}