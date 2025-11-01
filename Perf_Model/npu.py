from dataclasses import dataclass, field, asdict
import math
import threading
import numpy as np
import queue
import time

class Register:
    value: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class PE:

    reg0: np.uint32 = 0
    reg1: np.uint32 = 0
    regi: np.uint32 = 0

    def MAC(self, mac_input0, mac_input1) -> np.uint32:
        v0 = mac_input0.astype(np.uint32, copy=False)
        v1 = mac_input1.astype(np.uint32, copy=False)
        mac_out = (v0[0]*v1[0]) + (v0[1]*v1[1]) + (v0[2]*v1[2]) + (v0[3]*v1[3])
        self.reg1 = mac_out
    
    def MAC_subsequent(self, mac_input0, mac_input1):
        self.MAC(mac_input0, mac_input1)
        self.reg0 = self.reg1 + self.reg0
        
    def APPLY(self, bias)-> np.uint8:
        apply_out = bias + self.reg0
        return np.uint8(apply_out)
    
    def int32_to_int8(self, scale) -> np.uint8:
        quantizer_out = np.round(self.reg0/scale)
        quantizer_out = np.clip(quantizer_out, 0, 255)
        self.reg0 = quantizer_out.astype(np.uint8)

    def MAC_loop(self, buffer_in, end) -> np.uint8:
        self.regi = 0
        while(self.regi < end-1):
            if (self.regi == 0):
                self.MAC(buffer_in[0:4], buffer_in[4:8])
            else:
                self.MAC_subsequent(buffer_in[self.regi:self.regi+4], buffer_in[self.regi+4:self.regi+8])
            self.regi += 8
    
    def compute_node(self, buffer_in, end):
        self.MAC_loop(buffer_in, end)
        self.int32_to_int8(200)
        return self.APPLY(buffer_in[end])



class activation_function: 

    def relu6(self, activation_input) -> np.uint8:
        if (activation_input >= 6):
            activation_output = 6
        elif (activation_input < 0):
            activation_output = 0
        else:
            activation_output = activation_input
        return np.uint8(activation_output)

class pooling:

    def mean_pool() -> np.uint8:
        return 0


class NPU_compblock:

    def __init__(self):
        self.PE0 = PE()
        self.PE1 = PE()
        self.activation_func = activation_function()

        self.PE_buffer0 = np.zeros(30, dtype=np.uint8)
        self.PE_buffer1 = np.zeros(30, dtype=np.uint8)

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
        if (id == 0):
            self.PE_buffer0[:len(data)] = data
        elif (id == 1):
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
                break

            if task["target"] != pe_name:
                # Return task for the appropriate worker; both workers share the queue.
                self.task_queue.put(task)
                self.task_queue.task_done()
                continue
            end = task["end"]
            output = pe.compute_node(buffer, end)
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

    def execute_instruction(self, opcode, classifier, end):
        if opcode == 0:
            if classifier == 0:
                self.task_queue.put({"target": "PE0", "end": end})
            elif classifier == 1:
                self.task_queue.put({"target": "PE1", "end": end})
        elif opcode == 1:
            pass
        elif opcode == 4:
            self.stop_event.set()
            self.task_queue.put(None)
            self.output_queue.put(None)

    def get_activated_result(self, timeout=None):
        try:
            r = self.activation_out.get(timeout=timeout)
            self.activation_out.task_done()
            return r
        except queue.Empty:
            return None


# bit 0-2: instruction, bit 3: clasifier bit, 

if __name__ == "__main__":
    npu = NPU_compblock()

    in0 = np.arange(15, dtype=np.uint8)
    in1 = np.arange(12, dtype=np.uint8)
    npu.load_buffer(0, in0)
    npu.load_buffer(1, in1)

    npu.execute_instruction(0, 0, 10)
    npu.execute_instruction(0, 1, 9)

    print(npu.get_activated_result(timeout=2.0))
    print(npu.get_activated_result(timeout=2.0))

    npu.execute_instruction(4, 0, 0)
