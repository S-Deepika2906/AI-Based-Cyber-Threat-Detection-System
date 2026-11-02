# ============================================================
# THREAT SEVERITY ANALYSIS
# ============================================================

def calculate_severity(predicted_label, confidence, anomaly_status):
    """
    Calculate threat severity from:
    1. Predicted attack type
    2. XGBoost confidence
    3. Isolation Forest anomaly status

    Returns:
        severity_level
    """

    # --------------------------------------------------------
    # BENIGN TRAFFIC
    # --------------------------------------------------------

    if predicted_label == "Benign":

        if anomaly_status == "Anomaly":
            return "CRITICAL"

        return "LOW"


    # --------------------------------------------------------
    # KNOWN ATTACKS
    # --------------------------------------------------------

    if predicted_label == "PortScan":

        if confidence >= 0.95:
            return "HIGH"

        return "MEDIUM"


    if predicted_label == "FTP-Patator":

        return "HIGH"


    if predicted_label == "SSH-Patator":

        return "HIGH"


    if predicted_label == "Bot":

        return "HIGH"


    # --------------------------------------------------------
    # UNKNOWN ATTACK TYPE
    # --------------------------------------------------------

    return "CRITICAL"