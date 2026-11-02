# src/recommendations.py

def get_recommendations(predicted_label, hybrid_decision):
    """
    Return exactly 3 professional security recommendations
    based on the detected threat.
    """

    # ---------------------------------------------------------
    # Potential Novel / Suspicious Threat
    # ---------------------------------------------------------
    if "Potential Novel Threat" in hybrid_decision or "Suspicious" in hybrid_decision:
        return [
            "Investigate the anomalous traffic and correlate it with surrounding network events.",
            "Contain the suspected source or affected host using temporary network isolation or access controls.",
            "Escalate the event for analyst review and collect additional telemetry to determine whether it represents a new attack pattern."
        ]

    # ---------------------------------------------------------
    # PortScan
    # ---------------------------------------------------------
    if predicted_label == "PortScan":
        return [
            "Investigate the source IP for additional scanning activity and correlated connection attempts.",
            "Apply temporary network controls to restrict the identified source if the scanning activity persists.",
            "Review affected hosts and exposed services for unauthorized access attempts or follow-up exploitation."
        ]

    # ---------------------------------------------------------
    # FTP-Patator / SSH-Patator
    # ---------------------------------------------------------
    if predicted_label in ["FTP-Patator", "SSH-Patator"]:
        return [
            "Investigate the source IP and review authentication logs for repeated failed login attempts.",
            "Temporarily restrict the source or enforce rate limiting to contain brute-force activity.",
            "Verify affected accounts and strengthen authentication controls, including MFA where available."
        ]

    # ---------------------------------------------------------
    # Bot
    # ---------------------------------------------------------
    if predicted_label == "Bot":
        return [
            "Investigate the affected host for unusual outbound communication and suspected command-and-control activity.",
            "Isolate or restrict the affected host from non-essential network resources pending investigation.",
            "Collect endpoint and network indicators and perform malware or IOC analysis to identify related activity."
        ]

    # ---------------------------------------------------------
    # Benign
    # ---------------------------------------------------------
    if predicted_label == "Benign":
        return [
            "No immediate security action is required for this traffic.",
            "Continue monitoring for behavioral changes or repeated anomalies.",
            "Retain the event for baseline analysis and future threat correlation."
        ]

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------
    return [
        "Investigate the detected traffic and correlate it with related network activity.",
        "Apply appropriate temporary containment controls based on the observed risk.",
        "Escalate the event for further security analysis if suspicious behavior persists."
    ]