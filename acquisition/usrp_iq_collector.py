#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: USRP LoRa 实时 CRC 可视化采集器
# Author: OpenAI Codex
# Description: 采集前先实时解调并检查 CRC，确认位置可用后再打开正式录制开关。
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import gnuradio.lora_sdr as lora_sdr
import sip
import threading
import usrp_iq_collector_crc_packet_statistics as crc_packet_statistics  # embedded python block



class usrp_iq_collector(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "USRP LoRa 实时 CRC 可视化采集器", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("USRP LoRa 实时 CRC 可视化采集器")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_iq_collector")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.sync_word = sync_word = 0x12
        self.sf = sf = 12
        self.samp_rate = samp_rate = 2e6
        self.preamble_len = preamble_len = 16
        self.gain = gain = 20
        self.cr = cr = 4
        self.center_freq = center_freq = 487.7e6
        self.capture_session = capture_session = 0
        self.capture_run = capture_run = 0
        self.capture_root = capture_root = "../data/raw/ota"
        self.capture_location = capture_location = "lab1"
        self.capture_experiment = capture_experiment = 0
        self.capture_condition = capture_condition = "highsnr"
        self.bw = bw = 125e3
        self.settle_time = settle_time = 1
        self.rf_bandwidth = rf_bandwidth = 1e6
        self.record_enabled = record_enabled = False
        self.pay_len = pay_len = 33
        self.output_file = output_file = capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (capture_experiment, capture_session, capture_location, capture_condition, capture_run, sf, int(bw), int(samp_rate), preamble_len, sync_word, cr + 4, int(center_freq), ('%g' % gain).replace('.', 'p')))
        self.duration = duration = 780
        self.device_args = device_args = ''

        ##################################################
        # Blocks
        ##################################################

        self._record_enabled_choices = {'Pressed': bool(True), 'Released': bool(False)}

        _record_enabled_toggle_switch = qtgui.GrToggleSwitch(self.set_record_enabled, '正式录制（绿色=正在写盘）', self._record_enabled_choices, False, "green", "gray", 4, 50, 1, 1, self, 'value')
        self.record_enabled = _record_enabled_toggle_switch

        self.top_grid_layout.addWidget(_record_enabled_toggle_switch, 0, 0, 1, 1)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 1):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join((device_args, '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_bandwidth(rf_bandwidth, 0)
        self.uhd_usrp_source_0.set_rx_agc(False, 0)
        self.uhd_usrp_source_0.set_gain(gain, 0)
        self.uhd_usrp_source_0.set_min_output_buffer((int(samp_rate / bw * (2**sf + 2))))
        self.qtgui_sink_x_0 = qtgui.sink_c(
            4096, #fftsize
            window.WIN_BLACKMAN_hARRIS, #wintype
            center_freq, #fc
            samp_rate, #bw
            "USRP RX2 LoRa IQ", #name
            True, #plotfreq
            True, #plotwaterfall
            True, #plottime
            False, #plotconst
            None # parent
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(True)

        self.top_grid_layout.addWidget(self._qtgui_sink_x_0_win, 1, 0, 1, 4)
        for r in range(1, 2):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(0, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.qtgui_number_sink_crc = qtgui.number_sink(
            gr.sizeof_float,
            0,
            qtgui.NUM_GRAPH_NONE,
            5,
            None # parent
        )
        self.qtgui_number_sink_crc.set_update_time(0.2)
        self.qtgui_number_sink_crc.set_title("LoRa \u5B9E\u65F6 CRC \u8D28\u91CF")

        labels = ["\u672C\u5305 CRC", "\u5DF2\u89E3\u7801\u5305\u6570", "CRC \u6210\u529F\u6570", "\u7D2F\u8BA1 CRC \u6210\u529F\u7387", "\u6700\u8FD1 10 \u5305\u6210\u529F\u7387",
            "", "", "", "", ""]
        units = [" (1=PASS)", " packets", " packets", " %", " %",
            "", "", "", "", ""]
        colors = [("white", "black"), ("white", "black"), ("white", "black"), ("blue", "red"), ("blue", "red"),
            ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black"), ("black", "black")]
        factor = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

        for i in range(5):
            self.qtgui_number_sink_crc.set_min(i, 0)
            self.qtgui_number_sink_crc.set_max(i, 120)
            self.qtgui_number_sink_crc.set_color(i, colors[i][0], colors[i][1])
            if len(labels[i]) == 0:
                self.qtgui_number_sink_crc.set_label(i, "Data {0}".format(i))
            else:
                self.qtgui_number_sink_crc.set_label(i, labels[i])
            self.qtgui_number_sink_crc.set_unit(i, units[i])
            self.qtgui_number_sink_crc.set_factor(i, factor[i])

        self.qtgui_number_sink_crc.enable_autoscale(False)
        self._qtgui_number_sink_crc_win = sip.wrapinstance(self.qtgui_number_sink_crc.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._qtgui_number_sink_crc_win, 0, 1, 1, 3)
        for r in range(0, 1):
            self.top_grid_layout.setRowStretch(r, 1)
        for c in range(1, 4):
            self.top_grid_layout.setColumnStretch(c, 1)
        self.lora_sdr_header_decoder_0 = lora_sdr.header_decoder(False, int(cr), int(pay_len), True, 2, False)
        self.lora_sdr_hamming_dec_0 = lora_sdr.hamming_dec(True)
        self.lora_sdr_gray_mapping_0 = lora_sdr.gray_mapping( True)
        self.lora_sdr_frame_sync_0 = lora_sdr.frame_sync(int(center_freq), int(bw), int(sf), False, [sync_word], (int(samp_rate / bw)),int(preamble_len))
        self.lora_sdr_fft_demod_0 = lora_sdr.fft_demod( True, True)
        self.lora_sdr_dewhitening_0 = lora_sdr.dewhitening()
        self.lora_sdr_deinterleaver_0 = lora_sdr.deinterleaver( True)
        self.lora_sdr_crc_verif_0 = lora_sdr.crc_verif( 2, True)
        self.crc_packet_statistics = crc_packet_statistics.blk()
        self.blocks_skiphead_0 = blocks.skiphead(gr.sizeof_gr_complex*1, (int(settle_time * samp_rate)))
        self.blocks_skiphead_0.set_min_output_buffer((int(samp_rate / bw * (2**sf + 2))))
        self.blocks_null_sink_payload = blocks.null_sink(gr.sizeof_char*1)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, (int(duration * samp_rate)))
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, str(output_file), False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_copy_record = blocks.copy(gr.sizeof_gr_complex*1)
        self.blocks_copy_record.set_enabled(record_enabled)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.lora_sdr_header_decoder_0, 'frame_info'), (self.lora_sdr_frame_sync_0, 'frame_info'))
        self.connect((self.blocks_copy_record, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.blocks_copy_record, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.lora_sdr_frame_sync_0, 0))
        self.connect((self.blocks_skiphead_0, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.crc_packet_statistics, 2), (self.qtgui_number_sink_crc, 2))
        self.connect((self.crc_packet_statistics, 1), (self.qtgui_number_sink_crc, 1))
        self.connect((self.crc_packet_statistics, 3), (self.qtgui_number_sink_crc, 3))
        self.connect((self.crc_packet_statistics, 0), (self.qtgui_number_sink_crc, 0))
        self.connect((self.crc_packet_statistics, 4), (self.qtgui_number_sink_crc, 4))
        self.connect((self.lora_sdr_crc_verif_0, 0), (self.blocks_null_sink_payload, 0))
        self.connect((self.lora_sdr_crc_verif_0, 1), (self.crc_packet_statistics, 0))
        self.connect((self.lora_sdr_deinterleaver_0, 0), (self.lora_sdr_hamming_dec_0, 0))
        self.connect((self.lora_sdr_dewhitening_0, 0), (self.lora_sdr_crc_verif_0, 0))
        self.connect((self.lora_sdr_fft_demod_0, 0), (self.lora_sdr_gray_mapping_0, 0))
        self.connect((self.lora_sdr_frame_sync_0, 0), (self.lora_sdr_fft_demod_0, 0))
        self.connect((self.lora_sdr_gray_mapping_0, 0), (self.lora_sdr_deinterleaver_0, 0))
        self.connect((self.lora_sdr_hamming_dec_0, 0), (self.lora_sdr_header_decoder_0, 0))
        self.connect((self.lora_sdr_header_decoder_0, 0), (self.lora_sdr_dewhitening_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_skiphead_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "usrp_iq_collector")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_sync_word(self):
        return self.sync_word

    def set_sync_word(self, sync_word):
        self.sync_word = sync_word
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_sf(self):
        return self.sf

    def set_sf(self, sf):
        self.sf = sf
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))
        self.blocks_head_0.set_length((int(self.duration * self.samp_rate)))
        self.qtgui_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_preamble_len(self):
        return self.preamble_len

    def set_preamble_len(self, preamble_len):
        self.preamble_len = preamble_len
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))
        self.uhd_usrp_source_0.set_gain(self.gain, 0)

    def get_cr(self):
        return self.cr

    def set_cr(self, cr):
        self.cr = cr
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))
        self.qtgui_sink_x_0.set_frequency_range(self.center_freq, self.samp_rate)
        self.uhd_usrp_source_0.set_center_freq(self.center_freq, 0)

    def get_capture_session(self):
        return self.capture_session

    def set_capture_session(self, capture_session):
        self.capture_session = capture_session
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_capture_run(self):
        return self.capture_run

    def set_capture_run(self, capture_run):
        self.capture_run = capture_run
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_capture_root(self):
        return self.capture_root

    def set_capture_root(self, capture_root):
        self.capture_root = capture_root
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_capture_location(self):
        return self.capture_location

    def set_capture_location(self, capture_location):
        self.capture_location = capture_location
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_capture_experiment(self):
        return self.capture_experiment

    def set_capture_experiment(self, capture_experiment):
        self.capture_experiment = capture_experiment
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_capture_condition(self):
        return self.capture_condition

    def set_capture_condition(self, capture_condition):
        self.capture_condition = capture_condition
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_bw(self):
        return self.bw

    def set_bw(self, bw):
        self.bw = bw
        self.set_output_file(self.capture_root + ('/rxcap_exp%03d_sess%03d_loc%s_cond%s_run%03d_sf%d_bw%d_fs%d_pre%d_sw%02X_cr4%d_crc1_fc%d_rxg%s.cfile' % (self.capture_experiment, self.capture_session, self.capture_location, self.capture_condition, self.capture_run, self.sf, int(self.bw), int(self.samp_rate), self.preamble_len, self.sync_word, self.cr + 4, int(self.center_freq), ('%g' % self.gain).replace('.', 'p'))))

    def get_settle_time(self):
        return self.settle_time

    def set_settle_time(self, settle_time):
        self.settle_time = settle_time

    def get_rf_bandwidth(self):
        return self.rf_bandwidth

    def set_rf_bandwidth(self, rf_bandwidth):
        self.rf_bandwidth = rf_bandwidth
        self.uhd_usrp_source_0.set_bandwidth(self.rf_bandwidth, 0)

    def get_record_enabled(self):
        return self.record_enabled

    def set_record_enabled(self, record_enabled):
        self.record_enabled = record_enabled
        self.blocks_copy_record.set_enabled(self.record_enabled)

    def get_pay_len(self):
        return self.pay_len

    def set_pay_len(self, pay_len):
        self.pay_len = pay_len

    def get_output_file(self):
        return self.output_file

    def set_output_file(self, output_file):
        self.output_file = output_file
        self.blocks_file_sink_0.open(str(self.output_file))

    def get_duration(self):
        return self.duration

    def set_duration(self, duration):
        self.duration = duration
        self.blocks_head_0.set_length((int(self.duration * self.samp_rate)))

    def get_device_args(self):
        return self.device_args

    def set_device_args(self, device_args):
        self.device_args = device_args




def main(top_block_cls=usrp_iq_collector, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
