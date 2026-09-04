#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <string.h>

constexpr int STORAGE=200000; 
constexpr int BIT_OUT=128;


typedef ap_int<BIT_OUT> data_t;
typedef hls::axis<data_t, 0, 0, 0> axis_t;

void mm2s_replay_engine_V2(data_t *ddr_memory, int size, hls::stream<axis_t> &out_stream, bool enable) {
    // Interfaces
    #pragma HLS INTERFACE m_axi port=ddr_memory offset=slave bundle=gmem depth=10240 max_widen_bitwidth=BIT_OUT
    #pragma HLS INTERFACE s_axilite port=ddr_memory bundle=control
    #pragma HLS INTERFACE s_axilite port=size bundle=control
    #pragma HLS INTERFACE s_axilite port=enable bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    #pragma HLS INTERFACE axis port=out_stream
#pragma HLS PIPELINE II=1

    static int intern_counter = 0;
    static bool is_loaded = false;
    static data_t mem[STORAGE];
    
    #pragma HLS BIND_STORAGE variable=mem type=ram_2p impl=uram
	#pragma HLS AGGREGATE variable=mem compact=bit

    if (!enable) {
        intern_counter = 0;
        is_loaded = false;
        return;
    }


    if (enable && !is_loaded) {
        int copy_len = (size > STORAGE) ? STORAGE : size;
        memcpy(mem, (const data_t*)ddr_memory, copy_len * sizeof(data_t));
        is_loaded = true;
    }

    
    if (is_loaded) {
        for (int i = 0; i<size;i++) {
            #pragma HLS PIPELINE II=1

            axis_t output_packet;


            
            if (!enable) {
            	is_loaded = false;
                return;
            }
            
            output_packet.data = mem[intern_counter];

            // Side-Channels setzen (müssen definiert sein!)
            output_packet.keep = -1; // Alle Bytes gültig (0xF)
            output_packet.strb = -1;


            if (intern_counter == (size - 1)) {
                output_packet.last = 1;
            } else {
                output_packet.last = 0;
            }
            output_packet.keep = -1; // 0xFF...
            output_packet.strb = -1;
            
            out_stream.write(output_packet);


            
            intern_counter++;

            
            if (intern_counter >= size) {
                intern_counter = 0;

                
                #ifndef __SYNTHESIS__
                    break;
                #endif
            }
        }
    }
}
