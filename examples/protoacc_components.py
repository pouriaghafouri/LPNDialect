"""Composable building blocks for the simplified protoacc net."""

from __future__ import annotations

# Placeholders for injected names
create = None
emit = None
take = None
builder = None

import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional

from lpnlang_mlir import NetBuilder, PlaceHandle, Value
from examples.protoacc_helpers import CstStr, ProtoaccWorkload, load_workload
from examples.protoacc_weight_funcs import register_protoacc_weight_funcs

DEFAULT_WORKLOAD_PATH = Path.home(
) / "lpn-dev" / "lpn_examples" / "protoacc" / "hyperprotobench_processed" / "bench0-ser.json"


def _make_counter_token(count_value):
  token = create()
  token = token.set("remaining", count_value)
  return token


def _make_data_token(bytes_value,
                     *,
                     end_of_field=0,
                     end_of_top: Union[int, "Value"] = 0):
  token = create()
  token = token.set("bytes", bytes_value)
  token = token.set("end_of_field", end_of_field)
  token = token.set("end_of_top_level", end_of_top)
  return token


def _make_end_token(descriptor):
  token = _make_data_token(0, end_of_field=1, end_of_top=0)
  token = token.set("descriptor_id", descriptor.get("descriptor_id"))
  token = token.set("type", descriptor.get("type"))
  return token


def pass_all_tokens(src, dst, count):
  if isinstance(count, int):
    for _ in range(count):
      token = take(src)
      emit(dst, token)
  else:
    def loop_body(b, iv):
      token = b.take(src)
      b.emit(dst, token)
    builder.for_range(0, count, step=1, body=loop_body)


class DMAPortBase:
  def __init__(self, net: NetBuilder, name: str, direction: int, num_ports: int, width: int, latency: int, initiation_interval: int):
    self.net = net
    self.name = name
    self.ports = []
    for i in range(num_ports):
      put = net.place(f"{name}_put_{i}")
      get = net.place(f"{name}_get_{i}")
      fifo = net.place(f"{name}_fifo_{i}")
      self.ports.append((put, get, fifo, i))

  def port(self, idx: int) -> Tuple[PlaceHandle, PlaceHandle, PlaceHandle, int]:
    return self.ports[idx]


@dataclass
class FieldHandler:
  net: NetBuilder
  name: str
  handler_id: int
  # These will be connected by FrontEnd
  ptofieldhandler_dispatcher: Optional[PlaceHandle] = None
  pfields: Optional[PlaceHandle] = None
  pdispatch_hold: Optional[PlaceHandle] = None
  pwrite_hold: Optional[PlaceHandle] = None
  pwrites_input_IF_Q: Optional[PlaceHandle] = None
  
  mem_port: Optional[Tuple[PlaceHandle, PlaceHandle, PlaceHandle, int]] = None

  def __post_init__(self):
    # Places from components.py
    self.pops_in_ = self.net.place(f"{self.name}_pops_in_")
    self.p_dist_hold = self.net.place(f"{self.name}_p_dist_hold", initial_tokens=1)
    self.pops_in__eom = self.net.place(f"{self.name}_pops_in__eom")
    self.pops_in__scalar = self.net.place(f"{self.name}_pops_in__scalar")
    self.pops_in__non_scalar = self.net.place(f"{self.name}_pops_in__non_scalar")
    self.pops_in__repeated = self.net.place(f"{self.name}_pops_in__repeated")

    self.pdispatch_index_holder_ = self.net.place(f"{self.name}_pdispatch_index_holder_")

    self.p_S_WAIT_CMD = self.net.place(f"{self.name}_p_S_WAIT_CMD", initial_tokens=1)
    self.p_S_SCALAR_DISPATCH_REQ = self.net.place(f"{self.name}_p_S_SCALAR_DISPATCH_REQ")
    self.p_S_STRING_GETPTR = self.net.place(f"{self.name}_p_S_STRING_GETPTR")
    self.p_S_STRING_LOADDATA = self.net.place(f"{self.name}_p_S_STRING_LOADDATA")
    self.p_S_STRING_WRITE_KEY = self.net.place(f"{self.name}_p_S_STRING_WRITE_KEY")
    self.p_S_UNPACKED_REP_GETPTR = self.net.place(f"{self.name}_p_S_UNPACKED_REP_GETPTR")
    self.p_S_WRITE_KEY = self.net.place(f"{self.name}_p_S_WRITE_KEY")
    self.p_outputQ = self.net.place(f"{self.name}_p_outputQ")
    self.p_finished = self.net.place(f"{self.name}_p_finished")
    self.p_num_units = self.net.place(f"{self.name}_p_num_units")
    self.p_units = self.net.place(f"{self.name}_p_units")
    self.p_hold = self.net.place(f"{self.name}_p_hold", initial_tokens=1)

    self.pwrite_index_holder_ = self.net.place(f"{self.name}_pwrite_index_holder_")
    
    # Internal places for memory transitions
    self.p_t_36_pre = self.net.place(f"{self.name}_p_t_36_pre")
    self.p_t_30_pre = self.net.place(f"{self.name}_p_t_30_pre")
    self.p_t_37_pre = self.net.place(f"{self.name}_p_t_37_pre")
    self.p_t_37_pre2 = self.net.place(f"{self.name}_p_t_37_pre2")
    self.p_t_37_pre3 = self.net.place(f"{self.name}_p_t_37_pre3")
    self.p_t_44_pre = self.net.place(f"{self.name}_p_t_44_pre")

  def connect_memread_port(self, put_p, get_p, fifo_order_p, port_num):
    self.mem_port = (put_p, get_p, fifo_order_p, port_num)

  def build_transitions(self) -> None:
    helpers = register_protoacc_weight_funcs(self.net)
    pass_field_end_token = helpers.pass_field_end_token
    pass_key_outputQ_end_of_toplevel_token = helpers.pass_key_outputQ_end_of_toplevel_token
    pass_non_field_end_token = helpers.pass_non_field_end_token
    pass_write_hold_cond = helpers.pass_write_hold_cond
    pass_write_index_holder_cond = helpers.pass_write_index_holder_cond

    # tdispatch: ptofieldhandler_dispatcher, pdispatch_index_holder_, pfields -> ...
    @self.net.transition(f"{self.name}_dispatch")
    def tdispatch(ptofieldhandler_dispatcher=self.ptofieldhandler_dispatcher,
                  pdispatch_index_holder_=self.pdispatch_index_holder_,
                  pfields=self.pfields,
                  pdispatch_hold=self.pdispatch_hold,
                  pops_in_=self.pops_in_,
                  p_units=self.p_units,
                  p_num_units=self.p_num_units):
        token = take(ptofieldhandler_dispatcher)
        take(pdispatch_index_holder_)
        num = helpers.take_num_field_tokens(token)
        
        token_array = helpers.take_all_tokens(pfields, num)
        
        emit(pdispatch_hold, create())
        emit(pops_in_, token)
        
        for i in range(num):
          emit(p_units, token_array[i])
        
        emit(p_num_units, create().set("num", num))

    # t_resume: p_num_units, p_finished -> p_S_WAIT_CMD, p_outputQ
    @self.net.transition(f"{self.name}_resume")
    def t_resume():
        token = take(self.p_num_units)
        num = helpers.take_resume_token(token)
        helpers.take_all_tokens(self.p_finished, num)
        
        helpers.pass_empty_token(self.p_S_WAIT_CMD)
        helpers.pass_field_end_token(self.p_outputQ)

    # t_dist: pops_in_, p_dist_hold -> split
    @self.net.transition(f"{self.name}_dist")
    def t_dist():
        take(self.pops_in_)
        take(self.p_dist_hold)
        
        helpers.pass_eom(self.pops_in_, self.pops_in__eom)
        helpers.pass_scalar(self.pops_in_, self.pops_in__scalar)
        helpers.pass_non_scalar(self.pops_in_, self.pops_in__non_scalar)
        helpers.pass_repeated(self.pops_in_, self.pops_in__repeated)

    # t_eom
    @self.net.transition(f"{self.name}_eom")
    def t_eom():
      take(self.pops_in__eom)
      take(self.p_S_WAIT_CMD)
      take(self.p_units)
        
      helpers.pass_key_outputQ_end_of_toplevel_token(self.pops_in__eom, self.p_outputQ, 2.0)
        
      helpers.pass_empty_token(self.p_finished)
      helpers.pass_empty_token(self.p_dist_hold)

    # t_25 (scalar)
    @self.net.transition(f"{self.name}_t25")
    def t_25():
        take(self.pops_in__scalar)
        take(self.p_S_WAIT_CMD)
        helpers.pass_empty_token(self.p_S_SCALAR_DISPATCH_REQ, 1.0)
        helpers.pass_empty_token(self.p_dist_hold)

    # t_26 (non-scalar)
    @self.net.transition(f"{self.name}_t26")
    def t_26():
        take(self.pops_in__non_scalar)
        take(self.p_S_WAIT_CMD)
        helpers.pass_empty_token(self.p_S_STRING_GETPTR, 1.0)
        helpers.pass_empty_token(self.p_dist_hold)

    # t_28 (repeated)
    @self.net.transition(f"{self.name}_t28")
    def t_28():
        take(self.pops_in__repeated)
        take(self.p_S_WAIT_CMD)
        helpers.pass_token(self.pops_in__repeated, self.p_S_UNPACKED_REP_GETPTR, 1, 1.0)
        helpers.pass_empty_token(self.p_dist_hold)

    # t_31
    @self.net.transition(f"{self.name}_t31")
    def t_31():
        take(self.p_S_WRITE_KEY)
        emit(self.p_finished, create(), delay=1.0)
        helpers.pass_empty_token(self.p_hold)
        helpers.pass_key_outputQ_token(self.p_outputQ)

    # t_40
    @self.net.transition(f"{self.name}_t40")
    def t_40():
        take(self.p_S_STRING_WRITE_KEY)
        emit(self.p_finished, create(), delay=1.0)
        helpers.pass_key_outputQ_token(self.p_outputQ)
        helpers.pass_empty_token(self.p_hold)

    # t_write_req_out
    @self.net.transition(f"{self.name}_write_req_out")
    def t_write_req_out():
        take(self.p_outputQ)
        take(self.pwrite_index_holder_)
        
        helpers.pass_non_field_end_token(self.p_outputQ, self.pwrites_input_IF_Q, 1)
        helpers.pass_write_index_holder_cond(self.p_outputQ, self.pwrite_index_holder_, self.pwrite_index_holder_)
        helpers.pass_write_hold_cond(self.p_outputQ, self.pwrite_hold)

    # Memory transitions
    if self.mem_port:
        put_p, get_p, fifo_p, port_num = self.mem_port
        
        # t_30_pre
        @self.net.transition(f"{self.name}_t30_pre")
        def t_30_pre(p_S_SCALAR_DISPATCH_REQ=self.p_S_SCALAR_DISPATCH_REQ,
                     p_hold=self.p_hold,
                     p_t_30_pre=self.p_t_30_pre,
                     mem_put=put_p,
                     mem_fifo=fifo_p):
            take(p_S_SCALAR_DISPATCH_REQ)
            take(p_hold)
            helpers.pass_empty_token(p_t_30_pre, 2.0)
            
            helpers.mem_request(mem_put, port_num, CstStr.SCALAR_DISPATCH_REQ, 1)
            helpers.push_request_order(mem_fifo, port_num, 1)

        # t_30_post
        @self.net.transition(f"{self.name}_t30_post")
        def t_30_post(p_t_30_pre=self.p_t_30_pre,
                      mem_get=get_p,
                      p_units=self.p_units,
                      p_outputQ=self.p_outputQ,
                      p_S_WRITE_KEY=self.p_S_WRITE_KEY):
            take(p_t_30_pre)
            take(mem_get)
            take(p_units)
            helpers.pass_scalar_outputQ_token(p_units, p_outputQ)
            helpers.pass_empty_token(p_S_WRITE_KEY)

        # t_36_pre
        @self.net.transition(f"{self.name}_t36_pre")
        def t_36_pre(p_S_STRING_GETPTR=self.p_S_STRING_GETPTR,
                     p_hold=self.p_hold,
                     p_t_36_pre=self.p_t_36_pre,
                     mem_put=put_p,
                     mem_fifo=fifo_p):
            take(p_S_STRING_GETPTR)
            take(p_hold)
            helpers.pass_empty_token(p_t_36_pre, 4.0)
            
            helpers.mem_request(mem_put, port_num, CstStr.STRING_GETPTR_REQ, 3)
            helpers.push_request_order(mem_fifo, port_num, 3)

        # t_36_post
        @self.net.transition(f"{self.name}_t36_post")
        def t_36_post(p_t_36_pre=self.p_t_36_pre,
                      mem_get=get_p,
                      p_units=self.p_units,
                      p_S_STRING_LOADDATA=self.p_S_STRING_LOADDATA):
            take(p_t_36_pre)
            helpers.take_some_token(3) # Consumes from mem_get
            helpers.take_all_tokens(mem_get, 3) # Manual take because take_some_token returns int
            take(p_units)
            helpers.pass_token(p_units, p_S_STRING_LOADDATA, 1)

        # t_37_pre
        @self.net.transition(f"{self.name}_t37_pre")
        def t_37_pre(p_S_STRING_LOADDATA=self.p_S_STRING_LOADDATA,
                     p_t_37_pre=self.p_t_37_pre,
                     p_t_37_pre3=self.p_t_37_pre3,
                     mem_put=put_p,
                     mem_fifo=fifo_p):
            take(p_S_STRING_LOADDATA)
            helpers.pass_16_bytes_outputQ_token(p_S_STRING_LOADDATA, p_t_37_pre)
            helpers.pass_bytes_token(p_S_STRING_LOADDATA, p_t_37_pre3)
            
            helpers.mem_request_v2(p_S_STRING_LOADDATA, mem_put, port_num, CstStr.STRING_LOADDATA_REQ)
            helpers.push_request_order_v2(p_S_STRING_LOADDATA, mem_fifo, port_num)

        # t_37_post
        @self.net.transition(f"{self.name}_t37_post")
        def t_37_post(p_t_37_pre=self.p_t_37_pre,
                      mem_get=get_p,
                      p_outputQ=self.p_outputQ,
                      p_t_37_pre2=self.p_t_37_pre2):
            take(p_t_37_pre)
            take(mem_get)
            helpers.pass_token(p_t_37_pre, p_outputQ, 1)
            helpers.pass_empty_token(p_t_37_pre2)

        # t_37_post2
        @self.net.transition(f"{self.name}_t37_post2")
        def t_37_post2(p_t_37_pre3=self.p_t_37_pre3,
                       p_t_37_pre2=self.p_t_37_pre2,
                       p_S_STRING_WRITE_KEY=self.p_S_STRING_WRITE_KEY):
            token = take(p_t_37_pre3)
            num = helpers.get_num_from_token(token)
            helpers.take_all_tokens(p_t_37_pre2, num)
            
            helpers.pass_empty_token(p_S_STRING_WRITE_KEY)

        # t_44_pre
        @self.net.transition(f"{self.name}_t44_pre")
        def t_44_pre(p_S_UNPACKED_REP_GETPTR=self.p_S_UNPACKED_REP_GETPTR,
                     p_t_44_pre=self.p_t_44_pre,
                     mem_put=put_p,
                     mem_fifo=fifo_p):
            take(p_S_UNPACKED_REP_GETPTR)
            helpers.pass_token(p_S_UNPACKED_REP_GETPTR, p_t_44_pre, 1, 2.0)
            
            helpers.mem_request(mem_put, port_num, CstStr.UNPACKED_REP_GETPTR_REQ, 2)
            helpers.push_request_order(mem_fifo, port_num, 2)

        # t_44_post
        @self.net.transition(f"{self.name}_t44_post")
        def t_44_post(p_t_44_pre=self.p_t_44_pre,
                      mem_get=get_p,
                      p_S_SCALAR_DISPATCH_REQ=self.p_S_SCALAR_DISPATCH_REQ,
                      p_S_STRING_GETPTR=self.p_S_STRING_GETPTR):
            take(p_t_44_pre)
            helpers.take_some_token(2)
            helpers.take_all_tokens(mem_get, 2)
            
            helpers.pass_repeated_array_token(p_t_44_pre, p_S_SCALAR_DISPATCH_REQ, CstStr.SCALAR)
            helpers.pass_repeated_array_token(p_t_44_pre, p_S_STRING_GETPTR, CstStr.NONSCALAR)


class FrontEnd:
  def __init__(self, net: NetBuilder, name: str, num_handlers: int):
    self.net = net
    self.name = name
    self.num_handlers = num_handlers
    
    # Places from components.py
    self.pfields = net.place(f"{name}_pfields")
    self.pcontrol = net.place(f"{name}_pcontrol")
    self.pcontrol_prime = net.place(f"{name}_pcontrol_prime")
    self.pfields_meta = net.place(f"{name}_pfields_meta")
    self.pmessage_tasks = net.place(f"{name}_pmessage_tasks")
    self.ps_hasBitsLoader_HasBitsLoad = net.place(f"{name}_ps_hasBitsLoader_HasBitsLoad")
    self.pisnot_submessage_value_resp = net.place(f"{name}_pisnot_submessage_value_resp")
    self.pAdvance_OK = net.place(f"{name}_pAdvance_OK") # Initialized in bootstrap
    self.pdescr_request_Q = net.place(f"{name}_pdescr_request_Q")
    self.psWaitForRequest = net.place(f"{name}_psWaitForRequest") # Initialized in bootstrap
    self.ptofieldhandler_dispatcher = net.place(f"{name}_ptofieldhandler_dispatcher")
    self.pcollect = net.place(f"{name}_pcollect", observable=True)
    self.pholder_split_msg = net.place(f"{name}_pholder_split_msg") # Initialized in bootstrap
    self.p10_descr = net.place(f"{name}_p10_descr")
    self.p9_descr = net.place(f"{name}_p9_descr")

    self.pwrites_input_IF_Q = net.place(f"{name}_pwrites_input_IF_Q")
    self.pwrites_inject_Q = net.place(f"{name}_pwrites_inject_Q")
    self.pwrite_index_holder = net.place(f"{name}_pwrite_index_holder") # Initialized in bootstrap
    self.pwrite_mem_resp = net.place(f"{name}_pwrite_mem_resp")
    self.pwrites_inject_Q_non_top = net.place(f"{name}_pwrites_inject_Q_non_top")
    self.pwrites_inject_Q_top = net.place(f"{name}_pwrites_inject_Q_top")
    self.phold = net.place(f"{name}_phold") # Initialized in bootstrap
    self.pdispatch_index_holder = net.place(f"{name}_pdispatch_index_holder") # Initialized in bootstrap
    self.pdispatch_hold = net.place(f"{name}_pdispatch_hold") # Initialized in bootstrap
    self.pwrite_hold = net.place(f"{name}_pwrite_hold") # Initialized in bootstrap

    # Internal places for split transitions
    self.p_t_3_pre = net.place(f"{name}_p_t_3_pre")
    self.p_t_9_pre = net.place(f"{name}_p_t_9_pre")
    self.p10_descr_pre = net.place(f"{name}_p10_descr_pre")
    self.p10_descr_post = net.place(f"{name}_p10_descr_post")
    self.p10_descr_post2 = net.place(f"{name}_p10_descr_post2")
    self.p10_descr_post3 = net.place(f"{name}_p10_descr_post3")
    self.p_t_23 = net.place(f"{name}_p_t_23")

    self.handlers = {}
    self.mem_read_ports = []
    self.mem_write_port = None

  def connect_field_handler(self, handler: FieldHandler, idx: int):
    self.handlers[idx] = handler

  def connect_memread_port(self, put_list, get_list, fifo_list, port_num_list):
    self.mem_read_ports = list(zip(put_list, get_list, fifo_list, port_num_list))

  def connect_memwrite_port(self, put, get, fifo, port_num):
    self.mem_write_port = (put, get, fifo, port_num)

  def build_transitions(self):
    helpers = register_protoacc_weight_funcs(self.net)
    pass_not_submessage = helpers.pass_not_submessage
    pass_field_index_add_one = helpers.pass_field_index_add_one
    pass_token = helpers.pass_token
    push_request_order = helpers.push_request_order
    mem_request = helpers.mem_request

    # t1: pcontrol -> pdescr_request_Q, pisnot_submessage_value_resp
    @self.net.transition
    def t1():
        token = take(self.pcontrol)
        helpers.pass_token(token, self.pdescr_request_Q, 1)
        helpers.pass_not_submessage(token, self.pisnot_submessage_value_resp)

    # t2: pAdvance_OK, pmessage_tasks -> pcontrol
    @self.net.transition
    def t2():
        take(self.pAdvance_OK)
        task = take(self.pmessage_tasks)
        helpers.pass_token(task, self.pcontrol, 1, 1.0)

    # t3_pre: pisnot_submessage_value_resp -> p_t_3_pre, mem_req
    # mem_request(port_num, LOAD_HASBITS_AND_IS_SUBMESSAGE, 2)
    if self.mem_read_ports:
        put_0, get_0, fifo_0, port_num_0 = self.mem_read_ports[0]

        @self.net.transition
        def t3_pre(pisnot_submessage_value_resp=self.pisnot_submessage_value_resp,
                   p_t_3_pre=self.p_t_3_pre,
                   mem_put=put_0,
                   mem_fifo=fifo_0):
            token = take(pisnot_submessage_value_resp)
            helpers.pass_token(token, p_t_3_pre, 1, 1.0)
            
            helpers.push_request_order(mem_fifo, port_num_0, 2)
            helpers.mem_request(mem_put, port_num_0, CstStr.LOAD_HASBITS_AND_IS_SUBMESSAGE, 2)

        @self.net.transition
        def t3_post(p_t_3_pre=self.p_t_3_pre,
                    mem_get=get_0,
                    pAdvance_OK=self.pAdvance_OK,
                    ps_hasBitsLoader_HasBitsLoad=self.ps_hasBitsLoader_HasBitsLoad):
            take(p_t_3_pre)
            helpers.take_some_token(2)
            helpers.take_all_tokens(mem_get, 2)
            
            helpers.pass_empty_token(pAdvance_OK)
            helpers.pass_empty_token(ps_hasBitsLoader_HasBitsLoad)

        @self.net.transition
        def t9_pre(p9_descr=self.p9_descr,
                   psWaitForRequest=self.psWaitForRequest,
                   p_t_9_pre=self.p_t_9_pre,
                   mem_put=put_0,
                   mem_fifo=fifo_0):
            take(p9_descr)
            take(psWaitForRequest)
            helpers.pass_empty_token(p_t_9_pre)

            helpers.mem_request(mem_put, port_num_0, CstStr.LOAD_NEW_SUBMESSAGE, 2)
            helpers.push_request_order(mem_fifo, port_num_0, 2)

        @self.net.transition
        def t9_post(p_t_9_pre=self.p_t_9_pre,
                    mem_get=get_0,
                    psWaitForRequest=self.psWaitForRequest,
                    pAdvance_OK=self.pAdvance_OK,
                    pholder_split_msg=self.pholder_split_msg):
            take(p_t_9_pre)
            helpers.take_some_token(2)
            helpers.take_all_tokens(mem_get, 2)

            helpers.pass_empty_token(psWaitForRequest)
            helpers.pass_empty_token(pAdvance_OK)
            helpers.pass_empty_token(pholder_split_msg)

    # tload_field_addr: p10_descr_pre, ps_hasBitsLoader_HasBitsLoad, pfields_meta -> ...
    if len(self.mem_read_ports) > 1:
        put_1, get_1, fifo_1, port_num_1 = self.mem_read_ports[1]
        
        @self.net.transition
        def tload_field_addr(p10_descr_pre=self.p10_descr_pre,
                             ps_hasBitsLoader_HasBitsLoad=self.ps_hasBitsLoader_HasBitsLoad,
                             pfields_meta=self.pfields_meta,
                             p10_descr_post3=self.p10_descr_post3,
                             p10_descr_post=self.p10_descr_post,
                             mem_put=put_1,
                             mem_fifo=fifo_1):
            ctrl = take(p10_descr_pre)
            take(ps_hasBitsLoader_HasBitsLoad)
            
            count = helpers.take_num_field_as_control(ctrl)
            helpers.anonymous_func_1_pass_token(ctrl, p10_descr_post3)
            
            pass_all_tokens(pfields_meta, p10_descr_post, count)
            
            helpers.mem_request_v3(ctrl, mem_put, port_num_1, CstStr.LOAD_EACH_FIELD)
            helpers.push_request_order_v3(ctrl, mem_fifo, port_num_1)

        # tload_field_addr_post
        @self.net.transition
        def tload_field_addr_post(p10_descr_post=self.p10_descr_post,
                                  mem_get=get_1,
                                  ptofieldhandler_dispatcher=self.ptofieldhandler_dispatcher,
                                  p10_descr_post2=self.p10_descr_post2):
            meta = take(p10_descr_post)
            take(mem_get)
            
            helpers.pass_token(meta, ptofieldhandler_dispatcher, 1, 2.0)
            helpers.pass_empty_token(p10_descr_post2)

    # t10: p10_descr_post3, p10_descr_post2 -> pholder_split_msg
    @self.net.transition
    def t10():
        ctrl = take(self.p10_descr_post3)
        count = helpers.get_num_from_token(ctrl)
        
        helpers.take_all_tokens(self.p10_descr_post2, count)
        
        helpers.pass_empty_token(self.pholder_split_msg)

    # tsplit_msg: pdescr_request_Q, pholder_split_msg -> p10_descr_pre, p9_descr
    @self.net.transition
    def tsplit_msg():
        req = take(self.pdescr_request_Q)
        take(self.pholder_split_msg)
        
        helpers.pass_message_token(req, self.p9_descr)
        helpers.pass_non_message_token(req, self.p10_descr_pre)

    # t19: pwrites_input_IF_Q -> pwrites_inject_Q
    @self.net.transition
    def t19():
        token = take(self.pwrites_input_IF_Q)
        helpers.pass_token(token, self.pwrites_inject_Q, 1, 1.0)

    # tdist: pwrites_inject_Q, phold -> pwrites_inject_Q_non_top, pwrites_inject_Q_top
    @self.net.transition
    def tdist():
        token = take(self.pwrites_inject_Q)
        take(self.phold)
        
        helpers.pass_top_token(token, self.pwrites_inject_Q_top)
        helpers.pass_non_top_token(token, self.pwrites_inject_Q_non_top)

    # t24: pwrites_inject_Q_top -> pwrite_mem_resp, phold
    @self.net.transition
    def t24():
        take(self.pwrites_inject_Q_top)
        helpers.pass_empty_token(self.pwrite_mem_resp, 4.0)
        helpers.pass_empty_token(self.phold)

    # t23_pre: pwrites_inject_Q_non_top -> p_t_23, mem_req
    if self.mem_write_port:
        w_put, w_get, w_fifo, w_port = self.mem_write_port
        
        @self.net.transition
        def t23_pre(pwrites_inject_Q_non_top=self.pwrites_inject_Q_non_top,
                    p_t_23=self.p_t_23,
                    mem_put=w_put,
                    mem_fifo=w_fifo):
            token = take(pwrites_inject_Q_non_top)
            helpers.pass_empty_token(p_t_23)
            
            helpers.mem_request_write_v4(token, mem_put, w_port, CstStr.WRITE_OUT)
            helpers.push_write_request_order_v4(token, mem_fifo, w_port)

        @self.net.transition
        def t23_post(p_t_23=self.p_t_23,
                     phold=self.phold):
            take(p_t_23)
            helpers.pass_empty_token(phold)

    handler_targets = [self.handlers[idx] for idx in range(1, self.num_handlers + 1)]

    @self.net.transition(f"{self.name}_tdispatch_dist")
    def tdispatch_dist():
      idx_token = take(self.pdispatch_index_holder)
      take(self.pdispatch_hold)
      next_token = helpers.pass_field_index_add_one(idx_token, self.num_handlers)
      emit(self.pdispatch_index_holder, next_token)
      
      for idx, handler in self.handlers.items():
          helpers.pass_field_index_token(idx_token, handler.pdispatch_index_holder_, idx)

    @self.net.transition(f"{self.name}_twrite_dist")
    def twrite_dist():
      idx_token = take(self.pwrite_index_holder)
      take(self.pwrite_hold)
      next_token = helpers.pass_field_index_add_one(idx_token, self.num_handlers)
      emit(self.pwrite_index_holder, next_token)
      
      for idx, handler in self.handlers.items():
          helpers.pass_field_index_token(idx_token, handler.pwrite_index_holder_, idx)
        

def build_protoacc_net(workload: ProtoaccWorkload,
                       *,
                       num_handlers: int = 6,
                       name: str = "protoacc") -> NetBuilder:
  net = NetBuilder(name)
  
  # Create components
  dmaread = DMAPortBase(net, "dma_read_port", CstStr.READ, 8, 16, 20, 20)
  dmawrite = DMAPortBase(net, "dma_write_port", CstStr.WRITE, 1, 16, 20, 20)
  frontend = FrontEnd(net, "frontend", num_handlers)
  
  handlers: List[FieldHandler] = []
  for idx in range(num_handlers):
    handler = FieldHandler(net,
                           name=f"f{idx+1}",
                           handler_id=idx+1)
    
    # Connect shared places from FrontEnd
    handler.ptofieldhandler_dispatcher = frontend.ptofieldhandler_dispatcher
    handler.pfields = frontend.pfields
    handler.pdispatch_hold = frontend.pdispatch_hold
    handler.pwrite_hold = frontend.pwrite_hold
    handler.pwrites_input_IF_Q = frontend.pwrites_input_IF_Q
    
    handlers.append(handler)
    frontend.connect_field_handler(handler, idx+1)
    
    # Connect memread port (ports 2-7)
    if idx + 2 < 8:
        handler.connect_memread_port(*dmaread.port(idx + 2))

  # Connect frontend memread ports (ports 0, 1)
  memread_port_put_p_7, memread_port_get_p_7, memread_fifo_order_p_7, port_num_7 = dmaread.port(0)
  memread_port_put_p_8, memread_port_get_p_8, memread_fifo_order_p_8, port_num_8 = dmaread.port(1)

  frontend.connect_memread_port([memread_port_put_p_7, memread_port_put_p_8],
                                [memread_port_get_p_7, memread_port_get_p_8],
                                [memread_fifo_order_p_7, memread_fifo_order_p_8], 
                                [port_num_7, port_num_8])
  
  memwrite_port_put_p, memwrite_port_get_p, memwrite_fifo_order_p, port_num = dmawrite.port(0)
  frontend.connect_memwrite_port(memwrite_port_put_p, memwrite_port_get_p, memwrite_fifo_order_p, port_num)
  
  # Build transitions
  frontend.build_transitions()
  for handler in handlers:
      handler.build_transitions()

  descriptor_entries = list(workload.field_meta_tokens)
  unit_entries = list(workload.unit_tokens)
  seed = net.place("bootstrap_seed", initial_tokens=1)

  @net.transition
  def bootstrap(seed=seed):
    take(seed)
    
    # Initialize FrontEnd control tokens
    emit(frontend.pAdvance_OK, create())
    emit(frontend.psWaitForRequest, create())
    emit(frontend.pholder_split_msg, create())
    emit(frontend.phold, create())
    emit(frontend.pdispatch_hold, create())
    emit(frontend.pwrite_hold, create())
    
    idx_token = create().set("field_index", 1)
    emit(frontend.pdispatch_index_holder, idx_token.clone())
    emit(frontend.pwrite_index_holder, idx_token.clone())
    
    # Initialize FieldHandler control tokens
    handler_idx = 0
    while handler_idx < len(handlers):
        handler = handlers[handler_idx]
        emit(handler.p_dist_hold, create())
        emit(handler.p_S_WAIT_CMD, create())
        emit(handler.p_hold, create())
        handler_idx += 1

    # Populate workload
    control_entries = list(workload.control_tokens)
    ctrl_index = 0
    while ctrl_index < len(control_entries):
        entry = control_entries[ctrl_index]
        ctrl = create()
        ctrl = ctrl.set("type", int(entry.get("type", 0)))
        ctrl = ctrl.set("control_range", int(entry.get("control_range", 0)))
        ctrl = ctrl.set("repeated", int(entry.get("repeated", 0)))
        emit(frontend.pmessage_tasks, ctrl)
        ctrl_index += 1

    unit_index = 0
    desc_index = 0
    while desc_index < len(descriptor_entries):
      entry = descriptor_entries[desc_index]
      # Populate FrontEnd input places
      descriptor = create()
      descriptor = descriptor.set("descriptor_id", desc_index)
      descriptor = descriptor.set("type", int(entry.get("type", 0)))
      descriptor = descriptor.set("num", int(entry.get("num", 0)))
      descriptor = descriptor.set("repeated", int(entry.get("repeated", 0)))
      emit(frontend.pfields_meta, descriptor)
      
      unit_count = int(entry.get("num", 0))
      max_available = max(0, len(unit_entries) - unit_index)
      available = unit_count if unit_count < max_available else max_available
      offset = 0
      while offset < available:
        unit_info = unit_entries[unit_index + offset]
        unit_token = create()
        unit_token = unit_token.set("bytes", int(unit_info.get("bytes", 0)))
        emit(frontend.pfields, unit_token)
        offset += 1
      remaining = unit_count - available
      while remaining > 0:
        unit_token = create()
        unit_token = unit_token.set("bytes", 0)
        emit(frontend.pfields, unit_token)
        remaining -= 1
      unit_index += unit_count
      desc_index += 1

  # Let's add a sink for dmawrite put queue to drain it and print
  pwrite_sink = net.place("pwrite_sink", observable=True)
  
  @net.transition
  def drain_write(queue=dmawrite.port(0)[0], sink=pwrite_sink):
      token = take(queue)
      emit(sink, token)

  @net.transition
  def drain_write_fifo(fifo=dmawrite.port(0)[2]):
      take(fifo)

  # Memory Responder for Read Ports
  for i in range(8):
      put, get, fifo, _ = dmaread.port(i)
      
      def make_responder(idx, put=put, get=get, fifo=fifo):
          @net.transition(name=f"mem_respond_{idx}")
          def mem_respond(put=put, get=get, fifo=fifo):
              req = take(put)
              order = take(fifo)
              count = order.get("count")
              
              def loop_body(b, iv, get=get):
                  emit(get, create(), delay=10.0) # Add some memory latency
                  
              builder.for_range(0, count, step=1, body=loop_body)
      
      make_responder(i)

  return net


def emit_protoacc_net(workload_path: Path,
                        *,
                        num_handlers: int) -> str:
  workload = load_workload(workload_path)
  net = build_protoacc_net(workload, num_handlers=num_handlers)
  return net.build()


def simulate_protoacc_net(workload_path: Path,
                          *,
                          num_handlers: int,
                          max_time: float,
                          debug: bool) -> float:
  workload = load_workload(workload_path)
  net = build_protoacc_net(workload, num_handlers=num_handlers)
  simulator = net.python_simulator()
  final_time = simulator.run(max_time=max_time, debug=debug)
  print(f"Simulation finished at t={final_time:.3f} ns")
  for place in ("frontend_pcollect", "pwrite_sink"):
    tokens = simulator.place_dicts(place)
    print(f"  place {place}: {len(tokens)} token(s)")
    for token in tokens[:10]:
      print(f"    token {token}")
  return final_time


def main(argv: Optional[list[str]] = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workload",
                      type=Path,
                      default=DEFAULT_WORKLOAD_PATH,
                      help="Path to hyperprotobench_processed/*.json workload")
  parser.add_argument("--num-handlers",
                      type=int,
                      default=6,
                      help="Number of field handlers to instantiate")
  parser.add_argument("--simulate",
                      action="store_true",
                      help="Run the Python simulator instead of emitting MLIR")
  parser.add_argument("--max-time",
                      type=float,
                      default=200_000.0,
                      help="Simulation time budget (ns)")
  parser.add_argument("--debug",
                      action="store_true",
                      help="Enable verbose simulator tracing")
  args = parser.parse_args(argv)

  workload_path = args.workload.expanduser()
  if args.simulate:
    simulate_protoacc_net(workload_path,
                          num_handlers=args.num_handlers,
                          max_time=args.max_time,
                          debug=args.debug)
  else:
    mlir_text = emit_protoacc_net(workload_path,
                                  num_handlers=args.num_handlers)
    print(mlir_text)


if __name__ == "__main__":
  main()
