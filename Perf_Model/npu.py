from dataclasses import dataclass, field, asdict
import math
import threading
import numpy as np
import queue
import time
from enum import Enum


class instructions(Enum):
    MAC = 0b000
    LOAD = 0b001
    END = 0b011


mem = []
q = queue.Queue()
mem_counter = 0
offset = 0

REG_FILE = [0.0] * 63


@dataclass
class PE:

    reg0: float = 0.0
    reg1: float = 0.0
    regi: float = 0.0

    def MAC(self, mac_input0, mac_input1) -> float:
        v0 = mac_input0.astype(float, copy=False)
        v1 = mac_input1.astype(float, copy=False)
        mac_out = (v0[0] * v1[0]) + (v0[1] * v1[1]) + (v0[2] * v1[2]) + (v0[3] * v1[3])
        self.reg1 = mac_out

    def MAC_subsequent(self, mac_input0, mac_input1):
        self.MAC(mac_input0, mac_input1)
        self.reg0 = self.reg1 + self.reg0

    def APPLY(self, bias) -> float:
        apply_out = bias + self.reg0
        return float(apply_out)

    def MAC_loop(self, buffer_in, end) -> float:
        self.reg0 = 0.0
        self.reg1 = 0.0
        self.regi = 0

        max_len = min(end, len(buffer_in))

        while self.regi + 8 <= max_len:
            if self.regi == 0:
                self.MAC(buffer_in[0:4], buffer_in[4:8])
                self.reg0 = self.reg1
            else:
                self.MAC_subsequent(
                    buffer_in[self.regi:self.regi + 4],
                    buffer_in[self.regi + 4:self.regi + 8]
                )
            self.regi += 8

    def compute_node(self, buffer_in, end, bias):
        self.MAC_loop(buffer_in, end)
        return self.APPLY(10.27058)


class activation_function:

    def relu6(self, activation_input) -> float:
        if activation_input >= 6:
            activation_output = 6
        elif activation_input < 0:
            activation_output = 0
        else:
            activation_output = activation_input
        return float(activation_output)


class pooling:

    def mean_pool(self) -> float:
        return 0


class NPU_compblock:

    def __init__(self):
        self.PE0 = PE()
        self.PE1 = PE()
        self.activation_func = activation_function()

        self.PE_buffer0 = np.zeros(100, dtype=float)
        self.PE_buffer1 = np.zeros(100, dtype=float)

        self.inflight = 0
        self.inflight_lock = threading.Lock()
        self.finished_event = threading.Event()

        self.task_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.activation_out = queue.Queue()

        self.thread_pe0 = threading.Thread(target=self.pe_worker, args=(self.PE0, self.PE_buffer0, "PE0"), daemon=True)
        self.thread_pe1 = threading.Thread(target=self.pe_worker, args=(self.PE1, self.PE_buffer1, "PE1"), daemon=True)
        self.thread_activation = threading.Thread(target=self.activation_worker, daemon=True)

        self.thread_pe0.start()
        self.thread_pe1.start()
        self.thread_activation.start()

    def load_buffer(self, id, data):
        if id == 0:
            self.PE_buffer0[:len(data)] = data
        elif id == 1:
            self.PE_buffer1[:len(data)] = data
        else:
            raise ValueError("Error id")

    def pe_worker(self, pe, buffer, pe_name):
        while not self.stop_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                self.task_queue.task_done()
                break

            if task["target"] != pe_name:
                self.task_queue.put(task)
                self.task_queue.task_done()
                continue

            end = task["end"]
            bias = task["bias"]
            output = pe.compute_node(buffer, end, bias)
            print(f"\noutput for {pe_name}: {output}\n")
            self.output_queue.put({"pe": pe_name, "out": output})

            self.task_queue.task_done()

    def activation_worker(self):
        while not self.stop_event.is_set():
            try:
                result = self.output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if result is None:
                self.output_queue.task_done()
                break
            activated = self.activation_func.relu6(result["out"])
            self.activation_out.put({"pe": result["pe"], "activated": activated})
            self.output_queue.task_done()

            with self.inflight_lock:
                self.inflight -= 1
                if self.inflight == 0 and self.stop_event.is_set():
                    self.finished_event.set()

    def execute_instruction(self, opcode, classifier, start_reg, mem_location, length_reg, bias_reg):
        if opcode == instructions.MAC.value:
            start = start_reg
            length = length_reg

            data = mem[start:start + length]

            if classifier == 0:
                self.load_buffer(0, data)
                with self.inflight_lock:
                    self.inflight += 1
                self.task_queue.put({"target": "PE0", "end": length, "bias": bias_reg})
            elif classifier == 1:
                self.load_buffer(1, data)
                with self.inflight_lock:
                    self.inflight += 1
                self.task_queue.put({"target": "PE1", "end": length, "bias": bias_reg})

        elif opcode == instructions.LOAD.value:
            if 0 <= mem_location < len(mem) and 0 <= start_reg < len(REG_FILE):
                REG_FILE[start_reg] = mem[mem_location]
        elif opcode == instructions.END.value:
            self.stop_event.set()
            self.task_queue.put(None)
            self.output_queue.put(None)

    def get_activated_result_nowait(self):
        try:
            r = self.activation_out.get_nowait()
            self.activation_out.task_done()
            return r
        except queue.Empty:
            return None

    def has_inflight(self):
        with self.inflight_lock:
            return self.inflight > 0


if __name__ == "__main__":

    # load mem
    with open("mem_init.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            mem_val = float(line)
            mem.append(mem_val)

    instr_words = []
    with open("instr.txt", "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            instr = int(line, 2)
            instr_words.append(instr)

    npu = NPU_compblock()
    i = 0

    end_seen = False
    num_instrs = len(instr_words)

    while i < num_instrs:
        instr = instr_words[i]
        i += 1

        opcode = (instr >> 29) & 0b111
        classifier_bit = (instr >> 28) & 0b1
        reg_val0 = (instr >> 22) & 0b111111
        reg_val1 = (instr >> 16) & 0b111111
        reg_val2 = (instr >> 10) & 0b111111
        reg_val3 = (instr >> 4) & 0b111111
        mem_location = instr & 0b1111111111111111111111

        print("instr:", bin(instr))
        print("opcode:", opcode, "classifier:", classifier_bit, "reg0:", reg_val0, "reg1:", reg_val1, "reg2:", reg_val2, "reg3:", reg_val3, "mem_loc:", mem_location)

        npu.execute_instruction(opcode, classifier_bit, reg_val0, mem_location, reg_val1, reg_val2)

        if opcode == instructions.END.value:
            end_seen = True

        while True:
            res = npu.get_activated_result_nowait()
            if res is None:
                break
            pe_name = res["pe"]
            activated_value = res["activated"]
            print(f"Done: {pe_name}= {activated_value}")

        if end_seen and not npu.has_inflight():
            break
