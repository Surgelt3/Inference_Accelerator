set_global_assignment -entity "custom_ip_add" -library "lib_custom_ip_add" -name IP_TOOL_NAME "altera_fp_functions"
set_global_assignment -entity "custom_ip_add" -library "lib_custom_ip_add" -name IP_TOOL_VERSION "18.1"
set_global_assignment -entity "custom_ip_add" -library "lib_custom_ip_add" -name IP_TOOL_ENV "mwpim"
set_global_assignment -library "lib_custom_ip_add" -name SPD_FILE [file join $::quartus(sip_path) "custom_ip_add.spd"]

set_global_assignment -library "lib_custom_ip_add" -name MISC_FILE [file join $::quartus(sip_path) "custom_ip_add_sim/custom_ip_add.vo"]
