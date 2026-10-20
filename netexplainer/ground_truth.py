"""Ground truth of the extended questions (Q10-Q15), computed with scapy.

Every function takes the list of packets and returns the answer (an ``int`` or
a ``str``) or ``None`` when the question does not apply to the trace (e.g. DNS
questions on a trace without DNS), in which case the question is not asked for
that trace. Answers are deterministic: ties are listed sorted and joined with
``" or "`` (ports numerically, IPs as strings).
"""

from collections import Counter, defaultdict

SCAN_THRESHOLD = 10  # distinct destination ports of a "port scan" host
SYN, RST, ACK = 0x02, 0x04, 0x10

Q_TOP_PORT = "Which TCP/UDP destination port receives the most packets in the trace?"
Q_SYN_SOURCES = (
    "How many distinct source IPs send TCP SYN packets without the ACK flag?"
)
Q_DNS_QUERIES = "How many DNS queries (requests to UDP port 53) are in the trace?"
Q_TOP_TALKER = "Which IP address sends the most bytes in the trace?"
Q_SCANNER = (
    f"Does any host send packets to more than {SCAN_THRESHOLD} distinct TCP/UDP "
    "destination ports (possible port scan)? Give its IP address (all of them "
    "separated by ' or ' if there are several) or 'none'."
)
Q_RST = "How many TCP packets have the RST flag set?"


def _ip(packet):
    for layer in ("IP", "IPv6"):
        if packet.haslayer(layer):
            return packet[layer]
    return None


def _l4(packet):
    for layer in ("TCP", "UDP"):
        if packet.haslayer(layer):
            return packet[layer]
    return None


def _top(counter: Counter):
    """Key(s) with the highest count, sorted and joined with ' or '."""
    best = max(counter.values())
    return " or ".join(
        str(k) for k in sorted(k for k, v in counter.items() if v == best)
    )


def top_port(packets):
    ports = Counter(l4.dport for p in packets if (l4 := _l4(p)) is not None)
    return _top(ports) if ports else None


def syn_sources(packets):
    if not any(p.haslayer("TCP") for p in packets):
        return None
    return len(
        {
            _ip(p).src
            for p in packets
            if p.haslayer("TCP")
            and _ip(p) is not None
            and int(p["TCP"].flags) & (SYN | ACK) == SYN
        }
    )


def dns_queries(packets):
    dns = [p for p in packets if p.haslayer("UDP") and p.haslayer("DNS")]
    dns = [p for p in dns if 53 in (p["UDP"].sport, p["UDP"].dport)]
    return sum(1 for p in dns if p["DNS"].qr == 0) if dns else None


def top_talker(packets):
    sent = Counter()
    for p in packets:
        if (ip := _ip(p)) is not None:
            # frame length as shown by tshark (not the possibly snapped length)
            sent[ip.src] += getattr(p, "wirelen", None) or len(p)
    return _top(sent) if sent else None


def scanners(packets):
    ports = defaultdict(set)
    for p in packets:
        if (ip := _ip(p)) is not None and (l4 := _l4(p)) is not None:
            ports[ip.src].add(l4.dport)
    found = sorted(ip for ip, s in ports.items() if len(s) > SCAN_THRESHOLD)
    return " or ".join(found) if found else "none"


def rst_packets(packets):
    if not any(p.haslayer("TCP") for p in packets):
        return None
    return sum(1 for p in packets if p.haslayer("TCP") and int(p["TCP"].flags) & RST)


def core_answer(question: str, packets, total_size: int, duration):
    """Ground truth of the core questions (the 9 of the paper); ``None`` if
    ``question`` is not one of them. ``total_size`` is the sum of ``len`` of the
    packets and ``duration`` the time between the first and last packet."""
    if question == "What is the total number of packets in the trace?":
        return len(packets)

    elif question == "How many unique communicators are present in the trace?":
        unique_communicators = set()
        for packet in packets:
            if packet.haslayer("IP"):
                unique_communicators.add(packet["IP"].src)
                unique_communicators.add(packet["IP"].dst)
            elif packet.haslayer("IPv6"):
                unique_communicators.add(packet["IPv6"].src)
                unique_communicators.add(packet["IPv6"].dst)
        return len(unique_communicators)

    elif (
        question
        == "What is the IP that participates the most in communications in the trace?"
    ):
        ip_count = {}
        for packet in packets:
            if packet.haslayer("IP"):
                src_ip = packet["IP"].src
                dst_ip = packet["IP"].dst
                ip_count[src_ip] = ip_count.get(src_ip, 0) + 1
                ip_count[dst_ip] = ip_count.get(dst_ip, 0) + 1
            elif packet.haslayer("IPv6"):
                src_ip = packet["IPv6"].src
                dst_ip = packet["IPv6"].dst
                ip_count[src_ip] = ip_count.get(src_ip, 0) + 1
                ip_count[dst_ip] = ip_count.get(dst_ip, 0) + 1
        if ip_count:
            max_count = max(ip_count.values())
            most_common_ips = [
                ip for ip, count in ip_count.items() if count == max_count
            ]
            most_common_ip = (
                " or ".join(most_common_ips)
                if len(most_common_ips) > 1
                else most_common_ips[0]
            )
        else:
            most_common_ip = "No IP communications found"
        return most_common_ip

    elif question == "What is the total size of transmitted bytes?":
        return total_size

    elif question == "What is the average size of packets in bytes?":
        average_size = total_size / len(packets) if packets else 0
        return average_size

    elif question == "What predominates in the capture: ICMP, TCP, or UDP?":
        protocol_count = {"ICMP": 0, "ICMPv6": 0, "TCP": 0, "UDP": 0}
        for packet in packets:
            if packet.haslayer("ICMP"):
                protocol_count["ICMP"] += 1
            elif packet.haslayer("ICMPv6"):
                protocol_count["ICMPv6"] += 1
            elif packet.haslayer("TCP"):
                protocol_count["TCP"] += 1
            elif packet.haslayer("UDP"):
                protocol_count["UDP"] += 1
        if sum(protocol_count.values()) == 0:
            predominant_protocol = "No ICMP, ICMPv6, TCP, or UDP packets found"
        else:
            predominant_protocol = max(protocol_count, key=protocol_count.get)
        return predominant_protocol

    elif question == "How long in seconds does the communication last?":
        return duration

    elif question == "What is the average number of packets sent per second?":
        average_packets_per_second = (
            len(packets) / duration
            if duration > 0
            else "There is only one packet in the trace, operation not possible"
        )
        return average_packets_per_second

    elif question == "What is the average bytes/s sent in the communication?":
        average_bytes_per_second = (
            total_size / duration
            if duration > 0
            else "There is only one packet in the trace, operation not possible"
        )
        return average_bytes_per_second
    return None


EXTENDED = {
    Q_TOP_PORT: top_port,
    Q_SYN_SOURCES: syn_sources,
    Q_DNS_QUERIES: dns_queries,
    Q_TOP_TALKER: top_talker,
    Q_SCANNER: scanners,
    Q_RST: rst_packets,
}
