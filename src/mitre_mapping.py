# ============================================================
# MITRE ATT&CK MAPPING
# ============================================================

MITRE_MAPPING = {

    "PortScan": {
        "mitre_id": "T1046",
        "technique": "Network Service Scanning",
        "tactic": "Discovery"
    },

    "FTP-Patator": {
        "mitre_id": "T1110",
        "technique": "Brute Force",
        "tactic": "Credential Access"
    },

    "SSH-Patator": {
        "mitre_id": "T1110",
        "technique": "Brute Force",
        "tactic": "Credential Access"
    },

    "Bot": {
        "mitre_id": "T1071",
        "technique": "Application Layer Protocol",
        "tactic": "Command and Control"
    },

    "Benign": {
        "mitre_id": "N/A",
        "technique": "No Threat Detected",
        "tactic": "N/A"
    }
}


def get_mitre_mapping(predicted_label):

    return MITRE_MAPPING.get(
        predicted_label,
        {
            "mitre_id": "Unknown",
            "technique": "Requires Analyst Mapping",
            "tactic": "Unknown"
        }
    )