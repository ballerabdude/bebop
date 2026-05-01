mod ble;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(ble::BleManager::default())
        .invoke_handler(tauri::generate_handler![
            ble::ble_scan,
            ble::ble_connect,
            ble::ble_disconnect,
            ble::ble_get_device_info,
            ble::ble_scan_wifi,
            ble::ble_set_wifi_credentials,
            ble::ble_get_wifi_status,
            ble::ble_get_robot_config,
            ble::ble_set_robot_config,
            ble::ble_get_app_status,
            ble::ble_control_app,
            ble::ble_set_app_image,
            ble::ble_trigger_ota,
            ble::ble_get_ota_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
