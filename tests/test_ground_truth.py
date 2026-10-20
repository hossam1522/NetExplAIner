"""Ground truth of the extended questions against a synthetic scapy pcap."""

from unittest.mock import patch

import pytest
from scapy.all import DNS, DNSQR, IP, TCP, UDP, Ether, IPv6, wrpcap

from netexplainer import ground_truth as gt
from netexplainer.dataset import Dataset


def _tcp(src, dst, dport, flags, payload=b"", sport=40000):
    return (
        Ether()
        / IP(src=src, dst=dst)
        / TCP(sport=sport, dport=dport, flags=flags)
        / payload
    )


def _dns(src, dst, qr):
    dns = DNS(qr=qr, qd=DNSQR(qname="example.org"))
    if qr:
        return Ether() / IP(src=dst, dst=src) / UDP(sport=53, dport=5000) / dns
    return Ether() / IP(src=src, dst=dst) / UDP(sport=5000, dport=53) / dns


@pytest.fixture
def packets():
    pkts = [_tcp("10.0.0.1", "10.0.0.9", 1000 + i, "S") for i in range(12)]  # scanner
    pkts += [_tcp("10.0.0.2", "10.0.0.9", 80, "S", sport=41000 + i) for i in range(3)]
    pkts += [_tcp("10.0.0.9", "10.0.0.2", 41000, "SA", sport=80)]
    pkts += [_tcp("10.0.0.9", "10.0.0.2", 41001, "R", sport=80)]
    pkts += [_tcp("10.0.0.2", "10.0.0.9", 80, "RA", b"x" * 500)]  # top talker
    pkts += [_dns("10.0.0.2", "10.0.0.53", 0) for _ in range(3)]
    pkts += [_dns("10.0.0.2", "10.0.0.53", 1) for _ in range(2)]
    pkts.append(Ether() / IPv6(src="fe80::1", dst="fe80::2") / UDP(dport=9999))
    return pkts


def test_extended_answers(packets):
    assert gt.top_port(packets) == "80"  # 3 + 1 packets vs 53 (3), others (1)
    assert gt.syn_sources(packets) == 2  # 10.0.0.1 and 10.0.0.2 (SYN-ACK excluded)
    assert gt.dns_queries(packets) == 3
    assert gt.top_talker(packets) == "10.0.0.2"
    assert gt.scanners(packets) == "10.0.0.1"
    assert gt.rst_packets(packets) == 2  # R and RA


def test_ties_are_sorted_and_not_applicable_is_none():
    udp = [Ether() / IP(src="1.1.1.2", dst="9.9.9.9") / UDP(dport=p) for p in (70, 8)]
    assert gt.top_port(udp) == "8 or 70"  # numeric order
    assert gt.dns_queries(udp) is None and gt.syn_sources(udp) is None
    assert gt.rst_packets(udp) is None
    assert gt.scanners(udp) == "none" and gt.top_port([]) is None
    assert gt.top_talker(udp) == "1.1.1.2"


def test_scanner_threshold_is_strict():
    ten = [_tcp("1.1.1.1", "2.2.2.2", 100 + i, "S") for i in range(10)]
    assert gt.scanners(ten) == "none"
    assert gt.scanners(ten + [_tcp("1.1.1.1", "2.2.2.2", 999, "S")]) == "1.1.1.1"


def test_dataset_question_sets(tmp_path, packets):
    pcap = tmp_path / "t.pcap"
    wrpcap(str(pcap), packets)
    questions = gt.__file__.replace("ground_truth.py", "data/questions.yaml")

    def load(question_set):
        with patch(
            "netexplainer.dataset.check_output", return_value=b"1\t0\ta\tb\tc\t1\t"
        ):
            return Dataset(str(pcap), questions, "big", question_set)

    core, ext, both = load("core"), load("extended"), load("all")
    assert list(ext.question_indexes.values()) == [10, 11, 12, 13, 14, 15]
    assert list(core.question_indexes.values()) == list(range(1, 10))
    assert len(both.question_indexes) == 15 and both.n_packets == len(packets)
    assert [ext.questions_answers[q] for q in ext.question_indexes] == [
        "80",
        2,
        3,
        "10.0.0.2",
        "10.0.0.1",
        2,
    ]
    assert list(gt.EXTENDED) == list(ext.questions_subquestions)  # yaml == constants


def test_dataset_skips_not_applicable_questions(tmp_path):
    pcap = tmp_path / "u.pcap"
    wrpcap(str(pcap), [Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / UDP(dport=7)])
    questions = gt.__file__.replace("ground_truth.py", "data/questions.yaml")
    with patch("netexplainer.dataset.check_output", return_value=b"1\t0\ta\tb\tc\t1\t"):
        ds = Dataset(str(pcap), questions, "big", "extended")
    # no DNS, no TCP: Q11, Q12 and Q15 are skipped, indexes are kept
    assert list(ds.question_indexes.values()) == [10, 13, 14]
    with pytest.raises(ValueError):
        Dataset(str(pcap), questions, "big", "nope")
